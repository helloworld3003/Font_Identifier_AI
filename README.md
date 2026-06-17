# Font Identifier AI 

An ultra-scale Computer Vision pipeline designed to identify the exact typography used in any raw image or screenshot from a library of 100,000+ fonts. 

Built using PyTorch Metric Learning, the state-of-the-art **ConvNeXt-Tiny** backbone, and FAISS (Facebook AI Similarity Search), this architecture extracts 256-dimensional embeddings from fonts and places them onto a unit hypersphere. By leveraging Dynamic RAM Rendering, Cross-Batch Memory, and Adaptive Inference Binarization, it delivers pristine Train/Test symmetry and absolute gold-standard accuracy.

## System Architecture

Because generating static images for 100,000+ fonts creates a severe I/O bottleneck and disk failure risk, this pipeline entirely relies on **Dynamic RAM Rendering**.

### 🔹 Phase 1: Data Sanitization (`clean_dump.py`)
Cleans the raw font dump before training.
- Uses `fonttools` to safely parse binary headers and drops corrupt or 0-byte `.ttf` files.
- Ensures every font contains the standard English/Numeric glyphs (A-Z, 0-9).
- Deduplicates the dataset via MD5 hashing to prevent margin collapse during metric learning.

### 🔹 Phase 2: Dynamic Training Pipeline (`train_virtual_epochs.py`)
Trains the ConvNeXt-Tiny backbone using `MultiSimilarityLoss` and `CrossBatchMemory`.
- **Dynamic DataLoader:** Loads TTFs directly into RAM and uses `Pillow` to draw alphanumeric strings on the fly.
- **Train/Test Symmetry:** Augments data heavily using `Albumentations` (Perspective, Rotation, Blur, Noise, Compression) and applies a custom OpenCV `adaptiveThreshold` simulation to perfectly mimic real-world binarized crops.
- **Cross-Batch Memory (XBM):** Queues 4,096 historical embeddings to allow the miner to find the absolute "hardest" contrasting fonts far outside the current batch of 64.
- **Virtual Epochs:** Decouples dataset size from epoch length. One epoch is exactly 10,000 batches, preventing OOM crashes on 8GB VRAM cards like the RTX 5050.

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

## 🛠️ Requirements & Setup

You will need a GPU with CUDA support for production training. This architecture is heavily optimized to fit inside an 8GB VRAM envelope.

```bash
# Create a virtual environment
python -m venv font_env
font_env\Scripts\activate # Windows
# source font_env/bin/activate   # Mac/Linux

# Install all dependencies including PyTorch, FAISS, timm, and OpenCV
pip install -r requirements.txt
```

*(Note: Ensure you place your all `.ttf` files inside a `ttf_files/` directory before running Phase 1).*

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
# This will run indefinitely until the Cosine Annealing scheduler and patience triggers early stopping.
python train_virtual_epochs.py
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
