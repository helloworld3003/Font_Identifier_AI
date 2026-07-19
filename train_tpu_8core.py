import os
import gc
import random
import string
import logging
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

# Bypass Kaggle's deadly 64MB /dev/shm limit to prevent PyTorch background workers from crashing!
mp.set_sharing_strategy('file_system')

import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont

from torch.utils.data import Dataset, DataLoader, Sampler

# --- PyTorch XLA Imports ---
# We only import the multiprocessing wrapper globally. 
# We MUST NOT import the C++ backend (xm) globally, or the parent process will steal the TPU lock!
import torch_xla.distributed.xla_multiprocessing as xmp

# Configure global logger (mostly disabled for workers, handled via xm.master_print)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

import math

# ==========================================
# STATIC XLA LOSS IMPLEMENTATION
# ==========================================
class ArcFaceSupConLoss(nn.Module):
    """
    XLA-Friendly ArcFace Supervised Contrastive Loss.
    Combines the global pairwise structure of SupCon with the strict mathematical 
    angular margin penalty of ArcFace. Forces a strict geometric margin between 
    highly similar font clusters, making the task significantly harder and preventing 
    the network from plateauing on superficial features.
    """
    def __init__(self, scale=30.0, margin=0.50):
        super(ArcFaceSupConLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold for numerical stability
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        # Static Matrix Multiplication (B x B) -> Cosine Similarities
        cosine = torch.matmul(features, features.T)
        
        # Apply ArcFace margin to positive pairs using trig identities (XLA safe)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1) + 1e-9)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Static Binary Mask (1 for same class, 0 for different)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Mask out self-contrast (diagonal = 0)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # For positive pairs, use penalized phi. For negative pairs, use standard cosine.
        similarity_matrix = (mask * phi) + ((1.0 - mask) * cosine)
        
        # Scale the matrix (equivalent to 1 / temperature)
        similarity_matrix = similarity_matrix * self.scale
        
        # Numerical stability for exp
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        
        # Minimize the negative log-likelihood
        return -mean_log_prob_pos.mean()

# ==========================================
# HYPERPARAMETERS & KAGGLE AUTO-DETECT
# ==========================================
# Check if running on Kaggle and auto-mount the dataset to prevent symlink errors
kaggle_input_dir = Path("/kaggle/input")
if kaggle_input_dir.exists():
    try:
        # Prefer the exact 'ttf_files' folder to avoid mistakenly grabbing secondary datasets
        target_dir = None
        for d in kaggle_input_dir.rglob("ttf_files"):
            if d.is_dir():
                target_dir = d
                break
                
        if target_dir:
            TTF_DIR = str(target_dir)
        else:
            first_ttf = next(kaggle_input_dir.rglob("*.ttf"))
            TTF_DIR = str(first_ttf.parent)
            
        print(f"Auto-detected Kaggle dataset at: {TTF_DIR}")
    except StopIteration:
        TTF_DIR = "ttf_files"
else:
    TTF_DIR = "ttf_files"

# IMPORTANT: BATCH_SIZE is per-core! 
# 64 batch size * 8 cores = 512 Effective Batch Size.
BATCH_SIZE = 64  
M_PER_CLASS = 4

EMBEDDING_SIZE = 512
VIRTUAL_EPOCH_BATCHES = 1250
MAX_EPOCHS = 50
LEARNING_RATE_BACKBONE = 2e-5
LEARNING_RATE_HEAD = 5e-4
PATIENCE = 15

# ==========================================
# 1. DATASET & AUGMENTATION
# ========================================== 
def simulate_adaptive_threshold(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    block_size = np.random.choice([7, 11, 15, 19])
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, int(block_size), 2
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def get_train_transforms():
    return A.Compose([
        A.Rotate(limit=8, p=0.4),
        A.Perspective(scale=(0.05, 0.09), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # Disabled simulate_adaptive_threshold to prevent OpenCV memory leaks in multiprocessing
        # A.Lambda(image=simulate_adaptive_threshold, p=0.4),
        A.InvertImg(p=0.2), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class VirtualEpochBatchSampler(Sampler):
    def __init__(self, num_classes, batch_size, m_per_class, num_batches, start_batch=0):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.m_per_class = m_per_class
        self.num_batches = num_batches
        self.classes_per_batch = self.batch_size // self.m_per_class
        self.start_batch = start_batch

    def __iter__(self):
        # By skipping the first start_batch iterations, the DataLoader instantly fast-forwards
        # to where it left off without wasting time loading old images!
        for _ in range(self.start_batch, self.num_batches):
            classes = np.random.choice(self.num_classes, self.classes_per_batch, replace=False)
            batch = []
            for c in classes:
                batch.extend([c] * self.m_per_class)
            yield batch

    def __len__(self):
        return self.num_batches - self.start_batch

class DynamicFontDataset(Dataset):
    def __init__(self, ttf_dir, transform=None):
        self.ttf_files = list(Path(ttf_dir).rglob("*.ttf")) + list(Path(ttf_dir).rglob("*.otf"))
        
        # Filter out Font Decompression Bombs
        blacklist = set()
        if os.path.exists("bomb_blacklist.txt"):
            with open("bomb_blacklist.txt", "r") as f:
                blacklist = set([line.strip() for line in f.readlines()])
        self.ttf_files = [f for f in self.ttf_files if str(f) not in blacklist]
        
        if len(self.ttf_files) == 0:
            raise ValueError(f"No font files found in {ttf_dir}")
        self.transform = transform
        
        # Only master node should print dataset stats
        # Import xm locally so the parent process doesn't accidentally initialize the TPU backend
        import torch_xla.core.xla_model as xm
        xm.master_print(f"Loaded {len(self.ttf_files)} unique font files into the dataset.")

    def __len__(self):
        return len(self.ttf_files)

    def generate_random_string(self, length=5):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))

    def __getitem__(self, idx):
        ttf_path = self.ttf_files[idx]
        try:
            text = self.generate_random_string(random.randint(4, 10))
            font_size = random.randint(40, 80)
            font = ImageFont.truetype(str(ttf_path), font_size)
            
            temp_image = Image.new("RGB", (1, 1), "white")
            temp_draw = ImageDraw.Draw(temp_image)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_w = max(1, bbox[2] - bbox[0])
            text_h = max(1, bbox[3] - bbox[1])
            
            # CRITICAL FIX: Font Metric Bomb Protection!
            # Corrupted TTF files can have broken internal tables stating a character is 50,000 pixels wide.
            # If we pass that to Image.new(), Pillow attempts to allocate 30GB of RAM instantly, causing SIGKILL.
            if text_w > 5000 or text_h > 5000:
                raise ValueError("Font Metric Bomb detected! Skipping font.")
            
            image = Image.new("RGB", (text_w, text_h), "white")
            draw = ImageDraw.Draw(image)
            draw.text((-bbox[0], -bbox[1]), text, font=font, fill="black")
            
            target_w, target_h = 256, 64
            scale = min(target_w / text_w, target_h / text_h)
            new_w = max(1, int(text_w * scale))
            new_h = max(1, int(text_h * scale))
            
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            final_image = Image.new("RGB", (target_w, target_h), "white")
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            final_image.paste(image, (x, y))
            
            # Explicitly clean up C-level Pillow resources to prevent memory leaks!
            del draw
            del temp_draw
            del font
            del temp_image
            del image
            
        except Exception:
            return self.__getitem__(random.randint(0, len(self.ttf_files) - 1))
            
        image_np = np.array(final_image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
            del augmented
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        del image_np
        del final_image
        return image_tensor, idx

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
from model import ConvNeXtFontEncoder

# ==========================================
# 3. MULTIPROCESSING TRAINING LOOP (8-CORES)
# ==========================================
def _mp_fn(index, flags):
    """
    This function is replicated 8 times and runs independently on each TPU core.
    """
    # Import C++ backend ONLY inside the child processes!
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    
    # CRITICAL FIX: Disable OpenCV multithreading inside workers to prevent DataLoader crashes on Kaggle
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    
    # 1. Acquire current TPU core device
    device = xm.xla_device()
    xm.master_print(f"Executing Deep Metric Learning on 8 Google Cloud TPU Cores.")
    
    transform = get_train_transforms()
    dataset = DynamicFontDataset(TTF_DIR, transform=transform)
    
    # 2. Construct Data Samplers
    # Every core independently samples its own batch of 64 images.
    batch_sampler = VirtualEpochBatchSampler(
        num_classes=len(dataset),
        batch_size=BATCH_SIZE,
        m_per_class=M_PER_CLASS,
        num_batches=VIRTUAL_EPOCH_BATCHES
    )
    
    # CRITICAL FIX: PyTorch DataLoader Worker Process Rotation.
    # By setting num_workers=2 and persistent_workers=False, PyTorch will automatically DESTROY
    # the background rendering processes at the end of every epoch (1250 batches). 
    # This completely flushes the 16GB C++ Pillow/FreeType cache memory leak that caused SIGKILL.
    dataloader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        num_workers=2, 
        persistent_workers=False,
        pin_memory=False # pin_memory is only for CUDA GPUs, it causes memory leaks on TPUs!
    )
    
    # 3. Model & Loss Setup
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE).to(device)
    
    # XLA CRITICAL FIX: The PyTorch Metric Learning library uses boolean indexing to dynamically 
    # extract pairs/triplets. This forced PyTorch XLA to invoke the C++ compiler every single batch,
    # causing a 300GB RAM explosion and instantaneous system death.
    # By using our custom ArcFaceSupConLoss, the similarity matrix is strictly 512x512 forever,
    # resulting in exactly 1 compile step and zero memory leaks!
    loss_func = ArcFaceSupConLoss(scale=30.0, margin=0.50).to(device)
    
    # 4. Optimizer Setup
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE_BACKBONE},
        {'params': model.he_block.parameters(), 'lr': LEARNING_RATE_HEAD},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE_HEAD},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 1
    start_batch = 0
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(script_dir, "checkpoint.pth")
    MODEL_PATH = os.path.join(script_dir, "best_model.pth")
    LOG_CSV_PATH = os.path.join(script_dir, "training_log.csv")
    
    if not os.path.exists(CHECKPOINT_PATH):
        xm.master_print(f"No checkpoint file found at '{CHECKPOINT_PATH}'.")
        if os.path.exists(MODEL_PATH):
            xm.master_print(f"Found 'best_model.pth'. Loading pre-trained weights to resume...")
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            except Exception as e:
                xm.master_print(f"Failed to load 'best_model.pth': {e}")
        else:
            xm.master_print("Starting training from scratch (Epoch 1)...")
    else:
        xm.master_print(f"Found checkpoint file '{CHECKPOINT_PATH}'. Resuming training...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # PyTorch XLA CRITICAL FIX: Explicitly move optimizer momentum tensors to TPU
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
                        
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Mid-Epoch Resume Logic
            resume_epoch = checkpoint.get('epoch', 1)
            resume_batch = checkpoint.get('batch_idx', 0)
            
            if resume_batch >= VIRTUAL_EPOCH_BATCHES:
                start_epoch = resume_epoch + 1
                start_batch = 0
                xm.master_print(f"Resumed from completed Epoch {resume_epoch}.")
            else:
                start_epoch = resume_epoch
                start_batch = resume_batch
                xm.master_print(f"Resumed MID-EPOCH from Epoch {start_epoch}, Batch {start_batch}.")
                
        except Exception as e:
            xm.master_print(f"CRITICAL WARNING: Checkpoint file is corrupted! ({e})")
            xm.master_print("Attempting to fallback to 'best_model.pth' to rescue model weights...")
            if os.path.exists(MODEL_PATH):
                try:
                    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
                    xm.master_print("Successfully rescued model weights from 'best_model.pth'! Optimizer state was lost, resuming as a warm restart.")
                except Exception as e2:
                    xm.master_print(f"Failed to load 'best_model.pth': {e2}")
            else:
                xm.master_print("No fallback model found. Starting from scratch.")
        
    # --- ROBUST CSV SELF-CHECK & RECOVERY ---
    import csv
    if xm.is_master_ordinal():
        if os.path.exists(LOG_CSV_PATH):
            xm.master_print("Checking existing CSV log for integrity...")
            valid_rows = []
            with open(LOG_CSV_PATH, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    valid_rows.append(header)
                for row in reader:
                    try:
                        row_epoch = int(row[0])
                        # If a crash happened mid-epoch, the CSV will contain garbage partial data 
                        # for an epoch that wasn't fully checkpointed. We strictly purge it here!
                        if row_epoch < start_epoch:
                            valid_rows.append(row)
                    except ValueError:
                        pass
                        
            with open(LOG_CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(valid_rows)
            xm.master_print(f"CSV Self-Check complete! Purged any corrupted data from epoch >= {start_epoch}.")
        else:
            with open(LOG_CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "batch", "loss", "timestamp"])
                
    # CRITICAL XLA SYNC: Block all 8 cores until the Master Core finishes rewriting the CSV!
    xm.rendezvous("csv_sync")
    
    # All 8 cores now independently parse the cleaned CSV to recover the historic best_loss.
    # This guarantees the script NEVER forgets its lowest loss even if checkpoint.pth is tampered with!
    if os.path.exists(LOG_CSV_PATH):
        csv_best_loss = float('inf')
        with open(LOG_CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    if len(row) > 2 and row[1] == "EPOCH_SUMMARY":
                        val = float(row[2])
                        if val < csv_best_loss:
                            csv_best_loss = val
                except Exception:
                    pass
        
        if csv_best_loss < best_loss:
            best_loss = csv_best_loss
            xm.master_print(f"Recovered superior best_loss from CSV ground truth: {best_loss:.4f}")
            
    xm.master_print("Starting XLA 8-Core Training Pipeline...")
    
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        batch_sampler = VirtualEpochBatchSampler(
            num_classes=len(dataset),
            batch_size=BATCH_SIZE,
            m_per_class=M_PER_CLASS,
            num_batches=VIRTUAL_EPOCH_BATCHES,
            start_batch=start_batch
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_sampler=batch_sampler, 
            num_workers=2, 
            persistent_workers=False,
            pin_memory=False
        )
        
        # Parallel Loader forces data directly onto the TPU matrix
        mp_device_loader = pl.MpDeviceLoader(dataloader, device)
        
        start_time = time.time()
        
        for relative_batch_idx, (images, labels) in enumerate(mp_device_loader):
            batch_idx = relative_batch_idx + start_batch
            
            optimizer.zero_grad()
            
            # XLA handles bfloat16 seamlessly under the hood
            embeddings = model(images)
            
            # CRITICAL FIX: Global Contrastive Syncing (Cross-Batch Memory)
            # Gather embeddings and labels from all 8 cores to massively inflate the batch size
            # from 64 (16 fonts) to 512 (128 fonts)! This forces the AI to learn deep features.
            global_embeddings = xm.all_gather(embeddings, dim=0)
            global_labels = xm.all_gather(labels, dim=0)
            
            loss = loss_func(global_embeddings, global_labels)
            
            loss.backward()
            
            # Use XLA Optimizer Step (This automatically syncs and averages gradients across all 8 cores!)
            xm.optimizer_step(optimizer)
            
            # Extract loss for logging (using .item() forces a graph sync, but required for logging)
            current_loss = loss.item()
            running_loss += current_loss
            
            if (batch_idx + 1) % 100 == 0:
                xm.master_print(f"Epoch {epoch}/{MAX_EPOCHS} | Batch {batch_idx + 1}/{VIRTUAL_EPOCH_BATCHES} | Loss: {current_loss:.4f}")
                if xm.is_master_ordinal():
                    with open(LOG_CSV_PATH, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, batch_idx + 1, f"{current_loss:.4f}", time.time()])
                        
            # Save Mid-Epoch Checkpoint every 250 batches so we don't lose progress if Kaggle crashes!
            if (batch_idx + 1) % 250 == 0:
                checkpoint_state = {
                    'epoch': epoch,
                    'batch_idx': batch_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'epochs_no_improve': epochs_no_improve
                }
                xm.save(checkpoint_state, CHECKPOINT_PATH)
                xm.master_print(f"Mid-Epoch Checkpoint Saved at Batch {batch_idx + 1}")
                        
            # Physically purge Python garbage collector
            del images
            del labels
            del embeddings
            del loss
            gc.collect()
        
        avg_loss = running_loss / (batch_idx + 1)
        
        # Destroy the MpDeviceLoader to free all cached batches in TPU RAM
        del mp_device_loader
        gc.collect()
        
        # CRITICAL XLA FIX: We MUST synchronize the average loss across all 8 cores!
        # If we don't, each core will have a slightly different loss, causing them to diverge at the 
        # 'if avg_loss < best_loss' check. If one core decides to save and another doesn't, the TPU instantly deadlocks!
        avg_loss = xm.mesh_reduce("loss_reduce", avg_loss, np.mean)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        xm.master_print(f"=== Epoch {epoch} Summary ===")
        xm.master_print(f"Average Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if xm.is_master_ordinal():
            with open(LOG_CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, "EPOCH_SUMMARY", f"{avg_loss:.4f}", time.time()])
        
        # Save complete training state (Must be executed by all cores, but xm.save manages writing safely)
        checkpoint_state = {
            'epoch': epoch,
            'batch_idx': VIRTUAL_EPOCH_BATCHES, # Mark epoch as completely finished
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'epochs_no_improve': epochs_no_improve
        }
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            xm.save(model.state_dict(), MODEL_PATH)
            
            checkpoint_state['best_loss'] = best_loss
            checkpoint_state['epochs_no_improve'] = epochs_no_improve
            xm.save(checkpoint_state, CHECKPOINT_PATH)
            xm.master_print(f"New best loss achieved! Saved best_model.pth and checkpoint.pth")
        else:
            epochs_no_improve += 1
            checkpoint_state['epochs_no_improve'] = epochs_no_improve
            xm.save(checkpoint_state, CHECKPOINT_PATH)
            xm.master_print(f"No improvement. Early stopping patience: {epochs_no_improve}/{PATIENCE}. Saved checkpoint.pth")
            
        # Reset start_batch for the next epoch
        start_batch = 0
            
        import sys
        sys.stdout.flush()
            
        if epochs_no_improve >= PATIENCE:
            xm.master_print(f"Early stopping triggered after {epoch} epochs!")
            break
            
    xm.master_print("Training Pipeline Complete.")

if __name__ == "__main__":
    # Pre-download ConvNeXt weights on the master thread to prevent 8-core race conditions
    print("Pre-downloading ConvNeXt weights to avoid XLA multiprocessing race conditions...")
    _ = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
    
    # Kaggle injects legacy TPU environment variables that violently conflict with modern PyTorch XLA PJRT
    if 'TPU_PROCESS_ADDRESSES' in os.environ:
        del os.environ['TPU_PROCESS_ADDRESSES']
    if 'TPU_NAME' in os.environ:
        del os.environ['TPU_NAME']
        
    flags = {}
    # Use nprocs=None (default) so PJRT automatically detects all 8 TPU cores
    xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method='spawn')
