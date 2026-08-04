<div align="center">
  
# 🔤 Font Identifier AI

### An Ultra-Scale Computer Vision Pipeline for Typography Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/tapomoysarkar/ttf-files-for-fonts)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Website](https://img.shields.io/badge/Website-Netlify-00C7B7?style=for-the-badge&logo=netlify)](https://font-identifier-ai.netlify.app/)

### **Creator:** Tapomoy Sarkar
GPU Lender: Srijit Mondal

> An ultra-scale Computer Vision pipeline designed to identify the exact typography used in any raw image or screenshot from a library of **185,700+ fonts**.

</div>

---

## 🚀 The Problem & Our USP

Standard font identification tools rely on OCR or simple heuristics that fail on custom or noisy images. They often require the entire font file to be loaded into memory simultaneously, or have a very limited database of generic fonts.

**Our Difference:** We implemented **Deep Metric Learning**, the state-of-the-art **ConvNeXt-Tiny** backbone, and the **ArcFace (Additive Angular Margin)** Penalty to mathematically force the AI to distinguish between highly similar fonts (e.g., Arial vs. Helvetica). We combined this with a highly optimized backend server that uses **FAISS Memory Mapping** to search a vast embedding space in milliseconds without OOM crashes, making it fully deployable on low-RAM cloud environments.

### 1. Dynamic RAM Rendering
Because generating static images for 185,000+ fonts creates a severe I/O bottleneck and disk failure risk, this pipeline relies entirely on **Dynamic RAM Rendering**. We load TTFs directly into RAM and use `Pillow` to draw alphanumeric strings on the fly during training.

### 2. Multi-Stage Inference Engine
Our inference pipeline isn't just a raw neural network. We inject an OpenCV adaptive thresholding step to cleanly binarize real-world crops *before* passing them to the AI, ensuring pristine Train/Test symmetry.

---

## 🏛️ Architecture & Flowchart

<div align="center">
  <img src="Documents/architecture_diagram.svg" alt="Architecture Diagram" width="80%">
</div>

```mermaid
flowchart TB
    subgraph Frontend_Dashboard [Frontend Online Dashboard]
        direction LR
        UI[Web UI Interface] -->|Upload Image & Crop| API_Call(FastAPI REST Call)
        API_Call --> |JSON Response| Display[Visual Output rendering]
    end
    
    subgraph Backend_Server [Backend FastAPI Server]
        direction TB
        Recv[Receive Image Array] --> Preprocess[OpenCV Adaptive Binarization]
        Preprocess --> Resize[Aspect-Ratio Preserving Resize]
        Resize --> ONNX[ONNX Runtime: ConvNeXt-Tiny]
        ONNX --> |512D Vector| L2[L2 Normalization]
    end
    
    subgraph AI_Engine [AI & Vector Search]
        direction TB
        FAISS[(FAISS IndexFlatIP \nMemory Mapped)] 
        L2 --> |Similarity Search| FAISS
        FAISS --> Top10[Return Top 10 Matches]
        Top10 --> Metadata[Map to Kaggle TTF]
    end

    API_Call --> Recv
    Metadata --> Display
```

---

## 🖥️ Online Dashboard & Backend Server

Our solution provides a highly interactive and intuitive Web Dashboard that seamlessly connects to our FastAPI backend. 

### Interactive Dashboard
A full frontend web application allows users to upload screenshots, draw bounding boxes around text, and instantaneously fetch typographic results. 

**🌐 [Try the Live Website (Netlify)](https://font-identifier-ai.netlify.app/)**


https://github.com/user-attachments/assets/f318a7c9-dfcb-41c5-9556-001097a7046a



<div align="center">
  <img src="Documents/dashboard_page_1.png" alt="Dashboard Page 1" width="80%">
  <br><br>
  <img src="Documents/dashboard_page_2.png" alt="Dashboard Page 2" width="80%">
</div>

### High-Performance Backend
The backend utilizes FastAPI and is capable of ingesting raw images and performing ultra-fast vector retrieval on local edge devices.

<div align="center">
  <img src="Documents/Backend_server.png" alt="Backend Server Output" width="80%">
</div>

---

## 📊 The Dataset

This model is trained on a massive, combined dataset of approximately **185,700 fonts**, making it one of the largest specialized Computer Vision datasets for typography ever assembled.

- 🆓 **Free & Open-Source Fonts (~69,700 `.ttf` files):** Scraped from massive public repositories including Google Fonts, 1001 Fonts, and DaFont. These cover the vast majority of web and open-source typography.
- 💎 **Premium Fonts (~116,000 `.ttf` files):** Secured from a comprehensive commercial typography data leak, providing the neural network with unprecedented exposure to paid, proprietary, and highly niche design agency fonts.

🔗 **[View the Dataset on Kaggle](https://www.kaggle.com/datasets/tapomoysarkar/ttf-files-for-fonts)**

---

## 🏆 Our Accomplishments

### 1. The 185,000-File Dataset Bottleneck (Kaggle Integration)
We conquered the massive I/O bottleneck of dealing with 185,000 individual TTF files by bypassing local storage limits. We zip the raw fonts, mount them via Kaggle Datasets on high-speed NVMe storage, and dynamically stream the data.

### 2. Multi-Processing FAISS Engine & Memory Management
We completely eliminated OOM crashes during inference and index building by introducing Memory Mapped (MMAP) FAISS indices and a multi-process orchestrator. The AI smoothly processes and searches millions of parameters in an 8GB VRAM envelope.

### 3. Static XLA Loss for Google Cloud TPUs
Standard PyTorch metric learning libraries use dynamic boolean indexing which causes memory leaks on TPUs. We abandoned dynamic libraries and wrote a custom strictly-static **Supervised Contrastive Loss**. It uses pure static matrix multiplication, resulting in exactly 1 compile step and zero leaks forever on Google Cloud TPU v3-8 VMs.

---

## 🛠️ Requirements & Setup

You will need a GPU with CUDA support or a Google TPU for production training. This architecture is heavily optimized to fit inside an 8GB VRAM envelope.

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
