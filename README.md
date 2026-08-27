# MHCAFNet++: Deep Learning Architecture for Single-Image Dehazing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the codebase for **MHCAFNet++** (Multi-Hierarchy Channel Attention Fusion Network), an advanced and computationally efficient deep learning model for single-image dehazing. The codebase is fully modularized for easy training, evaluation, and inference.

## 📸 Visual Results

Below is a comparison of hazy inputs, our MHCAFNet++ restored outputs, and the ground truth clear images.

![Dehazing Results](<img width="1439" height="2490" alt="Output" src="https://github.com/user-attachments/assets/07f66638-4600-4b42-b739-98a53815d24a" />)

# MHCAFNet++: Multi-Hierarchy Channel Attention Fusion Network for Image Dehazing

This repository contains the codebase for **MHCAFNet++**, an advanced and efficient deep learning model for single-image dehazing. The code is structured and modularized for ease of use, training, and testing.

## Dataset
This model was trained and evaluated on the **RESIDE-OUT** dataset. 
You can download the dataset directly from Kaggle here: 
🔗 **[RESIDE-OUT Dataset on Kaggle](https://www.kaggle.com/datasets/anshkgoyal/reside-out)**

## Pre-trained Weights
Pre-trained weights are available in the GitHub Releases section of this repository. 
Download `MHCAFNetPP_RESIDE_Final (1).pth` from the [Releases Page](../../releases) and place it in the project root to run inference without training from scratch.

## Benchmark Results (RESIDE OUT)

Our model achieves state-of-the-art results on the RESIDE OUT benchmark, heavily outperforming previous architectures.

| Model | PSNR (dB) | SSIM |
| --- | --- | --- |
| DCP | 19.13 | 0.8148 |
| AOD-Net | 20.29 | 0.8765 |
| DehazeNet | 22.46 | 0.8514 |
| GFN | 21.55 | 0.8444 |
| FFA-Net | 33.57 | 0.9840 |
| **Ours (MHCAFNet++)** | **35.42** | **0.9855** |

## Model Architecture & Complexity

```mermaid
graph TD
    %% Node Styling
    classDef inputNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    classDef encNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef decNode fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef attNode fill:#fff8e1,stroke:#fbc02d,stroke-width:2px,color:#000
    classDef outNode fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000

    %% Main Architecture Flow

    Input("🌫️ Hazy Input Image (3, H, W)"):::inputNode

    subgraph Encoder ["Feature Extraction (Encoder)"]
        E1("Level 1<br>Conv 3x3 + ReLU + CA Block<br>(64 Channels)"):::encNode
        E2("Level 2<br>Max Pool + Conv 3x3 + CA Block<br>(128 Channels)"):::encNode
        E3("Level 3<br>Max Pool + Conv 3x3 + CA Block<br>(256 Channels)"):::encNode
    end

    subgraph Decoder ["Feature Fusion (Decoder)"]
        D1("Up-Level 1<br>ConvTranspose + Conv 3x3<br>(128 Channels)"):::decNode
        D2("Up-Level 2<br>ConvTranspose + Conv 3x3<br>(64 Channels)"):::decNode
    end

    Final("Final Layer<br>1x1 Conv (3 Channels)<br>☁️ Predicted Haze Map"):::decNode
    Subtract{"Global Residual<br>Subtraction<br>(Input - Haze)"}:::outNode
    Output("☀️ Clean Dehazed Image"):::outNode

    %% Core Connections
    Input --> E1
    E1 --> E2
    E2 --> E3
    E3 --> D1
    D1 --> D2
    D2 --> Final

    %% Skip Connections
    E2 -. "Skip Connection (Concat)" .-> D1
    E1 -. "Skip Connection (Concat)" .-> D2

    %% Global Residual Connection
    Input -. "Identity Mapping" .-> Subtract
    Final --> Subtract
    Subtract --> Output

    %% INVISIBLE LINK TO FORCE VERTICAL STACKING %%
    Output ~~~ CA_In
    %% Channel Attention Block Zoom-in
    subgraph CABlock ["🔍 Zoom: Channel Attention (CA) Module"]
        direction LR
        CA_In("Input Features"):::attNode
        GAP("Global Avg Pool"):::attNode
        Lin("Linear Layers + ReLU + Sigmoid"):::attNode
        Mult{{"Element-wise<br>Multiply"}}:::attNode
        
        CA_In --> GAP
        GAP --> Lin
        Lin --> Mult
        CA_In --> Mult
    end
```

The network utilizes a modular Block structure with embedded **Channel Attention** mechanisms and is optimized using a **Hybrid Loss Function** (L1 + SSIM + LPIPS) to preserve structural integrity and perceptual quality.

| Metric | Value |
| --- | --- |
| Total params | 2,475,067 |
| Trainable params | 2,475,067 |
| Non-trainable params | 0 |
| Input size (MB) | 1.69 |
| Forward/backward pass size (MB) | 1362.39 |
| Params size (MB) | 9.44 |
| Estimated Total Size (MB) | 1373.52 |

## Setup Instructions

### 1. Install Dependencies
Clone the repository and install the required Python packages:
```bash
git clone [https://github.com/davidsure2006/MHCAFNetPP-Image-Dehazing.git](https://github.com/davidsure2006/MHCAFNetPP-Image-Dehazing.git)
cd MHCAFNetPP-Image-Dehazing
pip install -r requirements.txt


i think readme is incomplete tell me what else need to be included

## 📁 Repository Structure

```text
MHCAFNetPP-Image-Dehazing/
├── assets/              # Images for README
├── dataset.py           # PyTorch Dataset class for RESIDE-OUT
├── model.py             # MHCAFNet++ network architecture
├── train.py             # Training loop and loss functions
├── test.py              # Inference and evaluation script
├── requirements.txt     # Python dependencies
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites

* Python 3.8+
* NVIDIA GPU + CUDA Toolkit (Highly recommended for training/inference)

### 1. Install Dependencies

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/davidsure2006/MHCAFNetPP-Image-Dehazing.git
cd MHCAFNetPP-Image-Dehazing
pip install -r requirements.txt
```

### 2. Training the Model

To train the model from scratch on the RESIDE-OUT dataset:

```bash
python train.py --data_dir /path/to/dataset --batch_size 16 --epochs 30
```

### 3. Evaluation & Inference

To evaluate the model or run inference on new hazy images using the pre-trained weights:

```bash
python test.py --data_dir /path/to/test_images --weights "MHCAFNetPP_RESIDE_Final (1).pth"
```

## 🙏 Acknowledgements

* The **RESIDE-OUT dataset** creators and Kaggle contributor **anshkgoyal**.
* Inspired by recent advancements in **Channel Attention** and **Global Residual Learning** for image restoration.
