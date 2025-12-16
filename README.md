# FADAStain
1. Introduction

Virtual staining aims to synthesize immunohistochemical (IHC) images from hematoxylin and eosin (H&E) stained images to enable molecular-level tissue analysis without additional staining cost. However, unavoidable spatial misalignment between consecutive tissue sections significantly degrades fine-grained staining localization and pathological consistency.

To address this issue, we propose FADAStain, a dual-constrained framework that:

Aligns semantically critical biomarker-positive regions via biomarker-guided feature-aware contrastive learning.

Mitigates synthetic-to-real IHC domain shift through discrepancy-aware learning with adaptive regularization.

The method does not require pixel-level alignment or additional manual annotations.

2. Method Overview

FADAStain consists of two core learning components:

2.1 Biomarker-Guided Feature-Aware Contrastive Learning

Automatically identifies biomarker-positive regions from real IHC images via threshold-based segmentation.

Performs hierarchical feature alignment between real and synthesized IHC images using contrastive learning.

Establishes semantic correspondence without relying on pixel-level alignment.

2.2 Discrepancy-Aware Learning

Dynamically identifies regions with maximum feature-space discrepancy between synthetic and real IHC images.

Enforces localized constraints to reduce domain shift.

Applies adaptive decay to avoid over-regularization during late-stage training.

The two strategies are jointly optimized to preserve fine-grained pathological semantics while stabilizing training.


4. Environment Setup

The experimental environment follows the configuration used in the paper.

conda env create -f environment.yml
conda activate fadastain

5. Datasets

All experiments strictly follow the dataset settings described in Section 3.1 of the paper.

Public Datasets

BCI Dataset

MIST Dataset

Dataset details and download links:

BCI: https://bupt-ai-cz.github.io/BCI

MIST: https://github.com/lifangda01/AdaptiveSupervisedPatchNCE

No additional annotations are required beyond the provided H&E–IHC image pairs.

6. Data Organization
dataset/

├── trainA/        # H&E images

├── trainB/        # IHC images

├── valA/

└── valB/

8. Training

python train.py --gpu_ids xx --dataroot xx --name train --checkpoints_dir xx --model FADAStain --CUT_mode FastCUT --n_epochs 80 --n_epochs_decay 0 --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 1.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --unet_seg MIST_unet_seg --lambda_gp 10.0 --gp_weights [0.015625,0.03125,0.0625,0.125,0.25,1.0] --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 2 --load_size 1024 --crop_size 512 --preprocess resize_and_crop --flip_equivariance False --display_winsize 512 --update_html_freq 100 --save_epoch_freq 5 --display_id 0

8. Testing

python test.py  --gpu_ids xx --dataroot xx --name train --checkpoints_dir xx --model FADAStain --CUT_mode FastCUT --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 1.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256  --lambda_gp 10.0 --gp_weights [0.015625,0.03125,0.0625,0.125,0.25,1.0] --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 4 --load_size 1024 --crop_size 1024 --preprocess resize_and_crop --flip_equivariance False --display_winsize 512 --num_test 1000 --phase val --results_dir xx

## Acknowledgement
This repo is built upon [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation) and [Adaptive Supervised PatchNCE Loss (ASP)](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE) -->
