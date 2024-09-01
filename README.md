# Vessel-like Structure Rehabilitation Network with Graph Clustering (VSR-Net)

This repository contains the official implementation of **Vessel-like Structure Rehabilitation Network with Graph Clustering (VSR-Net)**, as presented in the paper submitted to *IEEE Transactions on Image Processing*.

## Overview

The morphologies of vessel-like structures, such as blood vessels and nerve fibres, play significant roles in disease diagnosis, e.g., Parkinson's disease. Although deep network-based refinement segmentation and topology-preserving segmentation methods recently have achieved promising results in segmenting vessel-like structures, they still face two challenges: (1) existing methods often have limitations in rehabilitating subsection ruptures in segmented vessel-like structures; (2) they are typically overconfident in predicted segmentation results. To tackle these two challenges, this paper attempts to leverage the potential of spatial interconnection relationships among subsection ruptures from the structure rehabilitation perspective. Based on this perspective, we propose a novel Vessel-like Structure Rehabilitation Network (VSR-Net) to both rehabilitate subsection ruptures and improve the model calibration based on coarse vessel-like structure segmentation results. VSR-Net first constructs subsection rupture clusters via a Curvilinear Clustering Module (CCM). Then, the well-designed Curvilinear Merging Module (CMM) is applied to rehabilitate the subsection ruptures to obtain the refined vessel-like structures. Extensive experiments on six 2D/3D medical image datasets show that VSR-Net significantly outperforms state-of-the-art (SOTA) refinement segmentation methods with lower calibration errors. Additionally, we provide quantitative analysis to explain the morphological difference between the VSR-Net's rehabilitation results and ground truth (GT), which are smaller compared to those between SOTA methods and GT, demonstrating that our method more effectively rehabilitates vessel-like structures.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Additional dependencies listed in `requirements.txt`

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
