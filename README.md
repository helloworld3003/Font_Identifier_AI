# Font Identifier AI

An end-to-end Computer Vision pipeline designed to identify the exact typography used in any raw image or screenshot. 

Built using PyTorch Metric Learning, ResNet18, and FAISS (Facebook AI Similarity Search), this architecture extracts 128-dimensional geometric embeddings from fonts and maps them onto a unit hypersphere. This allows the AI to perform sub-millisecond nearest-neighbor searches against thousands of fonts using Cosine Similarity.

## System Architecture

The pipeline is split into four distinct phases, plus an automated MLOps Continuous Training orchestrator.

### 🔹 Phase 1: Synthetic Data Pipeline (`phase1_data_pipeline.py`)
Generates the massive training dataset from raw `.ttf` files.
- Uses `Pillow` and `FontTools` to render clean text images.
- Applies aggressive real-world augmentations (blur, noise, geometric distortion, color jitter) using `Albumentations`.
- Automatically labels and structures the data for PyTorch.

### 🔹 Phase 2: Production Metric Learning (`train_production.py`)
Trains the ResNet18 backbone using Triplet Margin Loss.
- Dynamically chunks large datasets (e.g., 3,800+ fonts) into VRAM-friendly batches.
- Uses `BatchHardMiner` to isolate visually identical fonts (e.g., serif variants) and actively push their embeddings apart.
- Implements `CosineAnnealingLR` to stabilize cluster geometry over 100 epochs.
- Includes auto-checkpointing and progress tracking.

### 🔹 Phase 3: Database Indexing (`phase3_database_indexing.py`)
Builds the ultra-fast search database.
- Passes a "Canonical String" through the trained model to extract a pristine 128D average representation for every font.
- Indexes all embeddings into a highly optimized binary FAISS `.index` tree.
- Exports the ID-to-Font translation dictionary as `faiss_mapping.csv`.

### 🔹 Phase 4: Inference Engine (`phase4_inference.py`)
The user-facing prediction script.
- Uses `EasyOCR` to detect text regions and bounding boxes within a raw, uncropped image.
- Intelligently crops, pads, and resizes each detection to perfectly match the Phase 1 training canvas distribution.
- Queries the FAISS database to return the Top-5 closest typographic matches with percentage confidences.
- Features a **Visual Annotation Module** that physically draws the bounding boxes and Top-1 predictions onto the output image (`visual_result.png`).

### 🚀 CI/CT Orchestrator (`add_new_fonts.py`)
An automated "plug-and-play" script for seamlessly expanding the AI's knowledge base.
- Automatically generates augmented data for any new `.ttf` files dropped in the `new_fonts/` directory.
- **Catastrophic Forgetting Safeguard:** Mixes the new fonts with a random subset of 50 previously trained anchor fonts to protect the integrity of the existing vector space.
- Fine-tunes the network for 20 epochs using an ultra-low learning rate (`1e-4`).
- Silently rebuilds the FAISS database in the background.

---

## 🛠️ Requirements & Setup

You will need a GPU with CUDA support for production training.

```bash
# Clone the repository
git clone https://github.com/yourusername/font-identifier-ai.git
cd font-identifier-ai

# Create a virtual environment
python -m venv font_env
source font_env/Scripts/activate # Windows
# source font_env/bin/activate   # Mac/Linux

# Install PyTorch (with CUDA) and dependencies
# NOTE: Install PyTorch according to your local CUDA version first!
pip install -r requirements.txt
```

*(Note: Ensure you place your initial `.ttf` files inside a `ttf_files/` directory before running Phase 1).*

## 🧠 Usage

**1. Generate the Dataset:**
```bash
python phase1_data_pipeline.py
```

**2. Train the Model (Chunked for VRAM safety):**
```bash
# Repeat this until the script indicates all fonts are trained
python train_production.py --chunk_size 1000 --save_name best_metric_model.pth
```

**3. Build the FAISS Index:**
```bash
python phase3_database_indexing.py
```

**4. Run an Inference Prediction:**
```bash
python phase4_inference.py "path/to/your/screenshot.png"
```

**5. Add New Fonts Later:**
Place new `.ttf` files into the `new_fonts/` directory, then simply run:
```bash
python add_new_fonts.py
```
