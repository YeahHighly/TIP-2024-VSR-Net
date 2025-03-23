import torch
import os

graph_dir = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/DRIVE_AV/training/graph_data/"
files = sorted(os.listdir(graph_dir))

for file in files[:5]:  # 只查看前5个文件
    graphs = torch.load(os.path.join(graph_dir, file), weights_only=False)
    print(f"{file} contains {len(graphs)} graphs")
    
