# 🔤 Font Identifier AI

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/tapomoysarkar/ttf-files-for-fonts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> An ultra-scale Computer Vision pipeline designed to identify the exact typography used in any raw image or screenshot from a library of **185,700+ fonts**.

Built using **Deep Metric Learning**, the state-of-the-art **ConvNeXt-Tiny** backbone, and **FAISS** (Facebook AI Similarity Search), this architecture extracts 256-dimensional embeddings from fonts and places them onto a unit hypersphere. By leveraging Dynamic RAM Rendering, Static Supervised Contrastive Loss (for TPUs), and Adaptive Inference Binarization, it delivers pristine Train/Test symmetry and absolute gold-standard accuracy.

---

## 📊 The Dataset

This model is trained on a massive, combined dataset of approximately **185,700 fonts**, making it one of the largest specialized Computer Vision datasets for typography ever assembled.

- 🆓 **Free & Open-Source Fonts (~69,700 `.ttf` files):** Scraped from massive public repositories including Google Fonts, 1001 Fonts, and DaFont. These cover the vast majority of web and open-source typography.
- 💎 **Premium Fonts (~116,000 `.ttf` files):** Secured from a comprehensive commercial typography data leak, providing the neural network with unprecedented exposure to paid, proprietary, and highly niche design agency fonts.

🔗 **[View the Dataset on Kaggle](https://www.kaggle.com/datasets/tapomoysarkar/ttf-files-for-fonts)**

---

## 🏗️ System Architecture

Because generating static images for 185,000+ fonts creates a severe I/O bottleneck and disk failure risk, this pipeline entirely relies on **Dynamic RAM Rendering**.

### 🔹 Phase 1: Data Sanitization (`clean_dump.py`)
Cleans the raw font dump before training.
- Uses `fonttools` to safely parse binary headers and drops corrupt or 0-byte `.ttf` files.
- Ensures every font contains the standard English/Numeric glyphs (A-Z, 0-9).
- Deduplicates the dataset via MD5 hashing to prevent margin collapse during metric learning.

### 🔹 Phase 2: Dynamic Training Pipelines
Two specialized training scripts exist depending on your hardware:

#### GPU Training (`train_virtual_epochs.py`)
Optimized for Nvidia GPUs (e.g., Kaggle 2x T4).
- **Dynamic DataLoader:** Loads TTFs directly into RAM and uses `Pillow` to draw alphanumeric strings on the fly.
- **Train/Test Symmetry:** Augments data heavily using `Albumentations` (Perspective, Rotation, Blur).
- **Extreme Speedhacks:** Uses `cudnn.benchmark = True`, non-blocking asynchronous memory transfers, and maxed-out batch sizes (1024) to peg GPUs at 100% utilization.

#### TPU Training (`train_tpu_8core.py`)
Optimized for Google Cloud TPU v3-8 VMs.
- **Static XLA Loss:** Replaces dynamic boolean masking with a custom `StaticSupConLoss` (Supervised Contrastive Loss) to eradicate XLA graph recompilation memory leaks.
- **IPC Bypass:** Implements `mp.set_sharing_strategy('file_system')` to completely bypass Kaggle's deadly 64MB `/dev/shm` Docker limits, allowing high-speed multiprocessing.

### 🔹 Phase 3: Database Indexing (`build_index.py`)
Builds the ultra-fast FAISS memory index.
- Passes 5 canonical strings (e.g., `"AaBbCc"`, `"xyz123"`) through the trained ConvNeXt weights.
- Averages the 5 embeddings into a stable, pristine 256D vector.
- Indexes all representations into a highly optimized binary `faiss.IndexFlatIP` tree.

### 🔹 Phase 4: Inference Engine (`inference.py`)
The production evaluation tool.
- Accepts a real-world image and a list of bounding boxes (`xmin,ymin,xmax,ymax`).
- Injects an OpenCV `adaptiveThreshold` step to cleanly binarize the crop *before* passing it to the neural network.
- Queries FAISS to return the Top-1 closest typographic match in milliseconds.
- Features a visual output that draws red bounding boxes and solid label tabs displaying the predicted Font Name and Confidence percentage (`visual_result.png`).

---

## 🚀 Challenges & Technical Breakthroughs

Building an ultra-scale font identifier locally and on cloud infrastructure posed immense technical hurdles. Here is how we solved the core challenges:

### 1. The 185,000-File Dataset Bottleneck (Kaggle Integration)
**The Problem:** Extracting, managing, and training on 185,000+ `.ttf` files locally caused catastrophic OS-level file handle limits and extreme I/O slowness. 
**The Solution:** We zipped the raw fonts, pushed them to Kaggle Datasets via the Kaggle API, and wrote a Kaggle-compatible metadata script. The training script now automatically auto-detects Kaggle mounting paths (`/kaggle/input/...`) and directly accesses the raw files on high-speed cloud NVMe storage.

### 2. GPU Slowness & The Shift to TPUs
**The Problem:** Training a deep metric learning model on 69,000+ active font classes locally on a standard Nvidia GPU would have taken months to reach convergence.
**The Solution:** We migrated the architecture to Google Cloud TPUs (`train_tpu_8core.py`). By utilizing PyTorch XLA `xmp.spawn` and distributing the data loader perfectly across 8 TPU cores, we unlocked massive multiprocessing scale.

### 3. Memory Leakage During Training (OOM Crashes)
**The Problem:** The training script would randomly spike to 300GB RAM usage and crash the Kaggle kernel. This was caused by two massive hidden bugs:
1. **"Font Metric Bombs":** Certain corrupted fonts had wildly incorrect internal glyph dimensions, forcing the `Pillow` rasterizer to allocate colossal (e.g., 50,000 x 50,000 pixel) canvases in RAM, instantly causing OOM.
2. **XLA Recompilation:** Standard PyTorch metric learning libraries use dynamic boolean indexing to extract positive/negative pairs. This forced the PyTorch XLA C++ compiler to completely rebuild the TPU graph *every single batch*, causing catastrophic memory leaks.
**The Solution:** We implemented strict max-dimension clamping (`<= 5000px`) and `try/except` bounds checking in the rasterizer. For the TPU leak, we completely abandoned dynamic libraries and wrote a custom strictly-static Supervised Contrastive Loss that uses pure static matrix multiplication, resulting in exactly *1 compile step* and zero leaks forever.

### 4. Memory Leakage During Index Building
**The Problem:** Running `build_index.py` to embed 69,000 fonts sequentially caused PyTorch and CUDA to silently leak memory over thousands of iterations, eventually causing the system to deadlock and crash before completion.
**The Solution:** We architected a Multi-Processed Orchestrator. The main script batches the fonts into chunks of 1000 and spawns a completely isolated child process to embed them. When the chunk is done, the child process is terminated, instantly forcing the OS to reclaim 100% of the leaked RAM before the next chunk begins.

### 5. Inability to Distinguish Extremely Similar Fonts
**The Problem:** Standard Supervised Contrastive (SupCon) loss proved "too easy" for the AI. The model plateaued around a loss of 1.15, learning superficial global textures but completely failing to distinguish between highly similar fonts (like Arial vs Helvetica).
**The Solution:** We completely overhauled the loss function to implement the state-of-the-art **ArcFace (Additive Angular Margin)** Penalty. By forcing a brutal, strict geometric 28-degree margin between every font cluster, the AI was mathematically tortured into learning incredibly deep, highly discriminative stroke structures. This pushed the model's accuracy through the roof.

---

## 🛠️ Requirements & Setup

You will need a GPU with CUDA support or a Google TPU for production training. This architecture is heavily optimized to fit inside an 8GB VRAM envelope (or a Kaggle TPU).

```bash
# Create a virtual environment
python -m venv font_env
source font_env/bin/activate   # Mac/Linux
# font_env\Scripts\activate    # Windows

# Install all dependencies including PyTorch, FAISS, timm, and OpenCV
pip install -r requirements.txt
```

*(Note: Ensure you place your all `.ttf` files inside a `ttf_files/` directory before running Phase 1).*

---

## 🧠 Usage

**1. Clean and Deduplicate the Dataset:**
```bash
python clean_dump.py
```

**2. Generate Typography Metadata (Optional but Recommended):**
```bash
python generate_metadata.py
```

**3. Train the Model:**
```bash
# For Nvidia GPUs (e.g., RTX 5050, 2x T4, A100)
python train_virtual_epochs.py

# For Google Cloud TPUs (e.g., Kaggle TPU v3-8)
python train_tpu_8core.py
```

**4. Build the FAISS Index:**
```bash
# Run this once best_model.pth is successfully saved
python build_index.py
```

**5. Run an Inference Prediction:**
```bash
# Pass the original image and bounding boxes as xmin,ymin,xmax,ymax
python inference.py sample.jpg 100,100,300,200 400,100,500,200
```

---

## ☁️ Running on Kaggle (Exact Steps)

Because training on 185,000 fonts requires serious cloud infrastructure, here is the exact step-by-step code you can copy and paste into a Kaggle Notebook to execute the entire pipeline on a TPU v3-8 VM.

**Step 1: Clone the GitHub Repository**
```python
!git clone https://github.com/helloworld3003/Font_Identifier_AI.git
```

**Step 2: Install Required Libraries**
```python
!pip install faiss-cpu timm albumentations pytorch-metric-learning otf2ttf
```

**Step 3: Download the Large Model Weights (Git LFS)**
```python
!apt-get install -y git-lfs
!cd /kaggle/working/Font_Identifier_AI && git lfs install && git lfs pull
```

**Step 4: Pull Latest Updates (Optional)**
```python
!cd /kaggle/working/Font_Identifier_AI && git pull origin main
```

**Step 5: Start TPU Training**
```python
!cd /kaggle/working/Font_Identifier_AI && python train_tpu_8core.py
```

**Step 6: Build the FAISS Index**
```python
!cd /kaggle/working/Font_Identifier_AI && python build_index.py
```
