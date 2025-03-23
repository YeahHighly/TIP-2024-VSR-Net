# VSR-Net: Vessel-Like Structure Rehabilitation Network With Graph Clustering

## Installation

This code was developed on Ubuntu 18.04.6 LTS.  Please install Anaconda first.

[1] To create a Conda environment
```bash
conda create --name vessel python=3.10
```

[2] To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Preparation
Please place the dataset in the following format:

```plaintext
TIP-2024-VSR-Net
├── coarse_segmentation
├── dataset
│   ├── DRIVE_AV
│   │   ├── training 
│   │   │   ├── images
│   │   │   │   ├── 21_training.tif
│   │   │   │   ├── 22_training.tif
│   │   │   │   ├── .....
│   │   │   │   └── 40_training.tif
│   │   │   ├── vessel
│   │   │   │   ├── 21_training.png
│   │   │   │   ├── 22_training.png
│   │   │   │   ├── .....
│   │   │   │   └── 40_training.png
│   │   └── testing
│   │   │   ├── images
│   │   │   │   ├── 01_testing.tif
│   │   │   │   ├── 02_testing.tif
│   │   │   │   ├── .....
│   │   │   │   └── 20_testing.tif
│   │   │   ├── vessel
│   │   │   │   ├── 01_testing.png
│   │   │   │   ├── 02_testing.png
│   │   │   │   ├── .....
│   │   │   │   └── 20_testing.png
│   │   └── testing
│   └── OCTA-500
│   │   ├── OCTAFULL 
│   │   │   │   ├── 10001.bmp
│   │   │   │   ├── 10002.bmp
│   │   │   │   ├── .....
│   │   │   │   └── 10300.bmp
│   │   └── GroundTruth
│   │   │   │   ├── 10001.bmp
│   │   │   │   ├── 10002.bmp
│   │   │   │   ├── .....
│   │   │   │   └── 10300.bmp
└── vsrnet
```

## Coarse Segmentation Model Training
```
cd coarse_segmentation
```

Select the coarse segmentation network and dataset, for example CENet and DRIVE dataset:

```
python train_coarse_2d.py --netwrok cenet --dataset drive
```

The saved model snapshots are available in the `checkpoints` folder. To evaluate the quality of the coarse segmentation results, run `inference.py` and `evaluation.py`.

Next, run `create_coarse_segmentation.py` to generate the corresponding coarse segmentation results for further training of VSR-Net.

```
python create_coarse_segmentation.py --netwrok cenet --dataset drive
```

The coarse segmentation results for both the training and test sets need to be generated separately. By default, they are saved in the `coarse` folder, located at the same level as the images and masks.

## VSR-Net
```
cd ../vsrnet/
```
### Graph and Mapping Preprocessing
Please run the corresponding data preprocessing scripts to generate the graph data and rehabilitation data required for training VSR-Net. 
For DRIVE Dataset:
```
cd ../dataloader/
python processing_graph_mapping.py
```

### CCM and CMM Module Training
Training the CCM module. For example, CCM_Base on the DRIVE dataset:
```
python train_ccm.py --module ccm --dataset drive
```
Training the CMM module. For example, CMM_Base on the DRIVE dataset:
```
python train_cmm.py --module ccm --dataset drive
```

### Performing Inference Using the Entire VSR-Net Pipeline.

```
python evaluation_vsrnet.py --ccm_module ccm --cmm_module cmm --dataset drive
```

## Citation
```
@article{Ye2025vsrnet,
  author  = {Haili Ye and Xiao-Qing Zhang and Yan Hu and Huazhu Fu and Jiang Liu},
  title   = {VSR-Net: Vessel-Like Structure Rehabilitation Network With Graph Clustering},
  journal = {IEEE Transactions on Image Processing},
  volume  = {34},
  pages   = {1090--1105},
  year    = {2025}
}
```





 








