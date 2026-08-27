```bash
# MHCAFNet++: Deep Learning Architecture for Single-Image Dehazing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the codebase for **MHCAFNet++** (Multi-Hierarchy Channel Attention Fusion Network), an advanced and computationally efficient deep learning model for single-image dehazing. The codebase is fully modularized for easy training, evaluation, and inference.

## 📸 Visual Results

*(Below is a comparison of hazy inputs, our MHCAFNet++ restored outputs, and the ground truth clear images)*

![Dehazing Results](assets/dehazing_results.png)

## 📊 Benchmark Results (RESIDE-OUT)

Our model achieves state-of-the-art results on the RESIDE-OUT benchmark, heavily outperforming previous architectures in both PSNR and SSIM.

| Model | PSNR (dB) | SSIM |
| --- | --- | --- |
| DCP | 19.13 | 0.8148 |
| AOD-Net | 20.29 | 0.8765 |
| DehazeNet | 22.46 | 0.8514 |
| GFN | 21.55 | 0.8444 |
| FFA-Net | 33.57 | 0.9840 |
| **Ours (MHCAFNet++)** | **35.42** | **0.9855** |

## 🧠 Model Architecture & Complexity

The network utilizes a modular Block structure with embedded **Channel Attention** mechanisms. It is optimized using a **Hybrid Loss Function** ($L_1$ + SSIM + LPIPS) to preserve structural integrity and perceptual quality while maintaining a lightweight parameter count.

| Metric | Value |
| --- | --- |
| Total params | 2,475,067 |
| Trainable params | 2,475,067 |
| Non-trainable params | 0 |
| Input size (MB) | 1.69 |
| Forward/backward pass size (MB) | 1362.39 |
| Params size (MB) | 9.44 |
| Estimated Total Size (MB) | 1373.52 |

### Architecture Diagram
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
git clone [https://github.com/davidsure2006/MHCAFNetPP-Image-Dehazing.git](https://github.com/davidsure2006/MHCAFNetPP-Image-Dehazing.git)
cd MHCAFNetPP-Image-Dehazing
pip install -r requirements.txt

```

### 2. Dataset

This model was trained and evaluated on the **RESIDE-OUT** dataset.
You can download the dataset directly from Kaggle here:
🔗 **[RESIDE-OUT Dataset on Kaggle](https://www.kaggle.com/datasets/anshkgoyal/reside-out)**

### 3. Pre-trained Weights

Download `MHCAFNetPP_RESIDE_Final (1).pth` from the [Releases Page](https://www.google.com/search?q=../../releases) and place it in the project root to run inference without training from scratch.

### 4. Training the Model

To train the model from scratch on the RESIDE-OUT dataset:

```bash
python train.py --data_dir /path/to/dataset --batch_size 16 --epochs 30

```

### 5. Evaluation & Inference

To evaluate the model or run inference on new hazy images using the pre-trained weights:

```bash
python test.py --data_dir /path/to/test_images --weights "MHCAFNetPP_RESIDE_Final (1).pth"

```

## 🙏 Acknowledgements

* The **RESIDE-OUT** dataset creators and Kaggle contributor [anshkgoyal](https://www.google.com/search?q=https://www.kaggle.com/anshkgoyal).
* Inspired by recent advancements in Channel Attention and Global Residual Learning for image restoration.

## 📄 License

This project is licensed under the MIT License.
EOF

```

```
