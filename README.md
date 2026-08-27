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
