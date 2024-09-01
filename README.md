# Vessel-like Structure Rehabilitation Network with Graph Clustering (VSR-Net)

This repository contains the official implementation of **Vessel-like Structure Rehabilitation Network with Graph Clustering (VSR-Net)**, as presented in the paper published in *IEEE Transactions on Image Processing*.

## Overview

VSR-Net is designed to rehabilitate vessel-like structures by leveraging graph clustering techniques to address ruptures and ensure topological consistency. The network integrates both deep learning and graph-based approaches, making it highly effective in handling delicate subsections and preserving the continuity of vessel-like structures.

## Features

- **Curvilinear Node Modeling**: Converts vessel-like structure ruptures into graph nodes.
- **Graph Clustering Module (CCM)**: Utilizes GCNs for clustering ruptures and ensuring spatial coherence.
- **Curvilinear Merging Module (CMM)**: Merges clustered subsections into complete vessel structures.
- **Parallel Processing**: Optimized for parallel inference to improve computational efficiency.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Additional dependencies listed in `requirements.txt`

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
