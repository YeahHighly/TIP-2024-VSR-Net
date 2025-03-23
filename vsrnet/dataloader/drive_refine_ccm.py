import os
import torch
from torch.utils.data import Dataset, DataLoader

class DRIVE_CCM_Dataset(Dataset):
    def __init__(self, graph_dir):
        """Dataset for loading preprocessed graph data."""
        self.graph_dir = graph_dir
        self.graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        """Load graph and corresponding labels."""
        graph_path = os.path.join(self.graph_dir, self.graph_files[idx])
        data_list = torch.load(graph_path, weights_only=False)
        return data_list  # Return list of graphs


def collate_fn(batch):
    """Flatten the batched list of graphs into a single sequence"""
    flattened_batch = []
    for graph_list in batch:
        if isinstance(graph_list, list):
            flattened_batch.extend(graph_list)  
        else:
            flattened_batch.append(graph_list)
    return flattened_batch


def get_drive_ccm_dataloader(graph_dir, batch_size=4, shuffle=True):
    """Returns DataLoader for preprocessed graph data."""
    dataset = DRIVE_CCM_Dataset(graph_dir=graph_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == "__main__":
    graph_dir = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/DRIVE_AV/testing/graph_data/"
    train_loader = get_drive_ccm_dataloader(graph_dir, batch_size=2, shuffle=True)

    for batch in train_loader:
        print("\nGraph statistics in batch:")
        for i, graph in enumerate(batch):

            # print(graph.edge_index.shape)
            # print(f"Graph {i+1}:")
            # print(f"  Nodes: {graph.x.shape[0]}")
            # print(f"  Edges: {graph.edge_index.shape[1]}")
            # print(f"  Edge values shape: {graph.edge_attr.shape}")
            # if len(graph.x.shape) < 4:
            print(f"  Node feature length: {graph.x.shape}")
            # print(f"  Edge labels: {graph.edge_label}")
        break  
