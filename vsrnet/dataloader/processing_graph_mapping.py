import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from tqdm import tqdm
from skimage import morphology
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
import time
import argparse
np.random.seed(1024)

# ------------------------ Argument Parser for Hyperparameters ------------------------
parser = argparse.ArgumentParser(description="Graph Edge Classification")
parser.add_argument("--dataset", type=str, default='drive', help="'drive' or 'octa'")
parser.add_argument("--sets", type=str, default='training', help="'training' or 'testing'")
args = parser.parse_args()

# ------------------------ Directory Setup ------------------------
if args.dataset.lower() == 'drive':
    base_path = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/DRIVE_AV"
    # Define directories
    coarse_dir       = os.path.join(base_path, args.sets, "coarse")
    mask_dir         = os.path.join(base_path, args.sets, "vessel")
    image_dir        = os.path.join(base_path, args.sets, "images")
    graph_data_dir   = os.path.join(base_path, args.sets, "graph_data")
    rehabilitate_dir = os.path.join(base_path, args.sets, "rehabilitate")
elif args.dataset.lower() == 'octa':
    base_path = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/OCTA-500"
    coarse_dir       = os.path.join(base_path, "coarse")
    mask_dir         = os.path.join(base_path, "OCTAFULL")
    image_dir        = os.path.join(base_path, "OCTAFULL")
    graph_data_dir   = os.path.join(base_path, "graph_data")
    rehabilitate_dir = os.path.join(base_path, "rehabilitate")
else:
    raise ValueError("Unsupported dataset. Use 'drive' or 'octa'.")

# Create required output directories
os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(rehabilitate_dir, exist_ok=True)

# Initialize ResNet18 as feature extractor
resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = nn.Identity()  # Remove classification layer
resnet18.eval().cuda()

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Parameters
patch_size = 256
num_patches = 4
max_distance = 50  # Maximum distance threshold for edges


def split_into_patches(image):
    """Split an image into 4x4 patches"""
    h, w = image.shape[:2]
    ph, pw = h // num_patches, w // num_patches
    return [
        image[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
        for i in range(num_patches) for j in range(num_patches)
    ]

def get_bounding_box(mask):
    """Get bounding box coordinates of a binary mask"""
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] == 0:
        return None  # No foreground pixels
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max

def split_connected_components(labels_mask):
    """Convert labels_mask into a binary matrix [n, w, h], where each slice is a connected component."""
    unique_labels = np.unique(labels_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (label 0)
    
    binary_masks = [(labels_mask == label).astype(np.uint8) for label in unique_labels]
    
    return np.stack(binary_masks) if binary_masks else np.empty((0, *labels_mask.shape), dtype=np.uint8)


def assign_contours_to_clusters(overlap_matrix, threshold=10):
    """
    Assign each connected_c_ to the best matching connected_m_, ensuring unique assignments.

    Parameters:
        overlap_matrix (numpy.ndarray): A matrix where entry (i, j) represents the overlap between connected_m_[i] and connected_c_[j].
        threshold (int): Minimum required overlap to consider assignment.

    Returns:
        dict: Mapping of connected_m_ indices to lists of assigned connected_c_ indices.
    """
    num_m, num_c = overlap_matrix.shape
    assigned_c = set()  # Track assigned c indices
    m_clusters = {i: [] for i in range(num_m)}  # Dictionary to store assigned c for each m

    for j in range(num_c):  # Iterate over c regions
        best_m_idx = np.argmax(overlap_matrix[:, j])  # Find the best m for this c
        max_overlap = overlap_matrix[best_m_idx, j]

        if max_overlap >= threshold and j not in assigned_c:
            m_clusters[best_m_idx].append(j)  # Assign c to this m
            assigned_c.add(j)  # Mark c as assigned

    return m_clusters

def compute_distance_matrix(connected_c_list):
    """
    Compute the pairwise shortest Euclidean distance between connected components.
    
    Parameters:
        connected_c_list (list of np.ndarray): List of binary masks representing connected components.

    Returns:
        np.ndarray: Distance matrix (size N x N) where entry (i, j) represents the shortest distance between component i and j.
    """
    centroids = []

    for component in connected_c_list:
        # Find coordinates of all nonzero (foreground) pixels
        coords = np.column_stack(np.where(component > 0))

        if coords.shape[0] == 0:
            continue  # Skip empty masks
        
        # Compute centroid (mean of all pixel coordinates)
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)

    if len(centroids) == 0:
        return np.array([])  # Return empty array if no valid components

    centroids = np.array(centroids)  # Convert to NumPy array

    # Compute pairwise Euclidean distance
    distance_matrix = cdist(centroids, centroids, metric='euclidean')

    return distance_matrix

def get_centroids(connected_list):
    """
    Computes the centroids of connected components in the input binary masks.

    Parameters:
        connected_list (list of np.ndarray): List of binary masks representing connected components.

    Returns:
        np.ndarray: Array of shape (N, 2), where each row is (y, x) centroid coordinates.
    """
    centroids = []
    center_point = []
    
    for component in connected_list:
        coords = np.column_stack(np.where(component > 0))  # Extract foreground pixel coordinates
        if coords.shape[0] == 0:
            continue  # Skip empty masks
        
        centroid = np.mean(coords, axis=0)  # Compute centroid (mean of all pixel coordinates)
        centroids.append(centroid)
        center_point.append(coords[int(len(coords)//2)])

        # temp = component.copy() * 255
        # temp = cv2.circle(cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB), (int(centroid[1]), int(centroid[0])), 10, (0, 0, 255), -1)
        # cv2.imwrite("./test_component/test_component_"+str(np.random.randint(1000))+".png", temp)

    return np.array(centroids) if centroids else np.empty((0, 2)), np.array(center_point) if center_point else np.empty((0, 2))  # Return empty if no centroids found

def construct_graph(distance_matrix, connected_c_list, connected_m_list, m_patch, threshold=100, top_k=20):
    coordinates, center_points = get_centroids(connected_c_list)

    # Extract node features using ResNet18
    connected_list_tensor = [transform(cv2.cvtColor(connected, cv2.COLOR_GRAY2RGB)) for connected in connected_c_list]
    # connected_list_tensor = torch.from_numpy(np.array(connected_list_tensor)).cuda()

    # with torch.no_grad():
    #     node_features_total = resnet18(connected_list_tensor).detach().cpu()  # Shape: (N, feature_dim)
    node_features_total = torch.from_numpy(np.array(connected_list_tensor))
    # print(node_features_total.shape)

    num_nodes = distance_matrix.shape[0]
    
    if num_nodes == 0:
        return None  # No nodes to form a graph

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-6
    inverse_dist = 1.0 / (distance_matrix + epsilon)

    normalized_weights = inverse_dist

    # Generate edge indices, edge attributes, and labels
    edge_list = []
    edge_weights = []
    edge_labels = []
    node_position = []
    node_features = []
    
    edge_set = set()
    node_set = set()

    # output_dir = "./test_inter_graph/"
    for i in range(num_nodes):
        node_position.append((int(center_points[i][1]), int(center_points[i][0])))
        node_features.append(node_features_total[i])

        nearest_neighbors = np.argsort(distance_matrix[i])[1:top_k+1]  

        for j in nearest_neighbors:
            if i > j and ((i, j) not in edge_set or (j, i) not in edge_set) and distance_matrix[i][j] < threshold:
                union_connected_1 = np.sum(connected_c_list[i] * connected_m_list, axis=(1, 2))
                union_connected_2 = np.sum(connected_c_list[j] * connected_m_list, axis=(1, 2))
                union_connected = np.where(union_connected_1 * union_connected_2 > 0, True, False)

                edge_set.add((i, j))
                edge_set.add((j, i))
                edge_list.append((i, j))
                edge_weights.append(normalized_weights[i, j])

                if True in union_connected:
                    edge_labels.append(1)
                else:
                    edge_labels.append(0)

    # union_c = np.sum(connected_c_list, axis=0)
    # union_c = np.where(union_c > 0, 255, 0).astype(np.uint8)
    # visualize_graph = cv2.cvtColor(union_c, cv2.COLOR_GRAY2RGB)

    # clean_image = visualize_graph.copy()

    # gt_visualize_graph =  np.sum(connected_m_list, axis=0)
    # gt_visualize_graph = np.where(gt_visualize_graph > 0, 255, 0).astype(np.uint8)
    # gt_visualize_graph = cv2.cvtColor(gt_visualize_graph, cv2.COLOR_GRAY2RGB)

    # for i in range(num_nodes):
    #     visualize_graph = cv2.circle(visualize_graph, (int(center_points[i][1]), int(center_points[i][0])), 5, (0, 255, 0), -1)

    # for (ex, ey), el in zip(edge_list, edge_labels):
    #     if el == 1:
    #         start_pos = (int(center_points[ex][1]), int(center_points[ex][0]))
    #         end_pos = (int(center_points[ey][1]), int(center_points[ey][0]))
    #         visualize_graph = cv2.line(visualize_graph, start_pos, end_pos, (0, 0, 255), 2)
    #     # else:
    #     #     start_pos = (int(center_points[ex][1]), int(center_points[ex][0]))
    #     #     end_pos = (int(center_points[ey][1]), int(center_points[ey][0]))
    #     #     visualize_graph = cv2.line(visualize_graph, start_pos, end_pos, (255, 0, 0), 2)

    # visualize_graph = np.hstack((clean_image, visualize_graph, gt_visualize_graph))

    # random_filename = f"test_visualize_{np.random.randint(1000)}.png"
    # save_path = output_dir + random_filename
    # cv2.imwrite(save_path, visualize_graph)

    if len(edge_list) > 0: 
        edge_list = np.swapaxes(np.array(edge_list), 0, 1)

        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(edge_list, dtype=torch.long)  # Shape: (2, num_edges)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)  # Edge weights
        edge_label = torch.tensor(edge_labels, dtype=torch.long)  # Edge classification labels

        node_features = torch.tensor(np.array(node_features), dtype=torch.float32) #.reshape(-1, 512)
        node_position = torch.tensor(np.array(node_position), dtype=torch.long).reshape(-1, 2)

        # Create PyTorch Geometric Graph
        graph_data = Data(x=node_features, pos=node_position, edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)

        return graph_data

    else:
        return None

def process_image(coarse_path, mask_path, image_path, graph_dir, rehab_dir):
    """Process an image to extract graph data and rehabilitation segmentation"""
    coarse = cv2.imread(coarse_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(image_path)
    coarse_patches = split_into_patches(coarse)
    mask_patches = split_into_patches(mask)
    image_patches = split_into_patches(original_image)

    graphs = []
    
    for idx, (c_patch, m_patch, img_patch) in enumerate(zip(coarse_patches, mask_patches, image_patches)):
        
        kernel = np.ones((3,3),np.uint8)  
        c_patch = cv2.morphologyEx(c_patch, cv2.MORPH_CLOSE, kernel)
        m_patch = cv2.morphologyEx(m_patch, cv2.MORPH_CLOSE, kernel)

        c_patch = cv2.dilate(c_patch, kernel, iterations = 1)
        m_patch = cv2.dilate(m_patch, kernel, iterations = 1)
        
        contours_c, _ = cv2.findContours(c_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_m, _ = cv2.findContours(m_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # colored_c = np.zeros_like(img_patch)
        # for contour in contours_c:
        #     color = tuple(np.random.randint(0, 255, 3).tolist())  # Random RGB color
        #     cv2.drawContours(colored_c, [contour], -1, color, -1)

        # colored_m = np.zeros_like(img_patch)
        # for contour in contours_m:
        #     color = tuple(np.random.randint(0, 255, 3).tolist())  # Random RGB color
        #     cv2.drawContours(colored_m, [contour], -1, color, -1)

        # colored_m_path = os.path.join(rehab_dir, f"{os.path.basename(coarse_path).replace('.png', '')}_colorm_{idx}.png")
        # colored_c_path = os.path.join(rehab_dir, f"{os.path.basename(coarse_path).replace('.png', '')}_colorc_{idx}.png")
        
        # cv2.imwrite(colored_m_path, colored_m)
        # cv2.imwrite(colored_c_path, colored_c)

        connected_m_list = []
        for contour in contours_m:
            m_patch_bin = np.zeros_like(m_patch, dtype=np.uint8)
            cv2.drawContours(m_patch_bin, [contour], -1, 1, -1)
            connected_m_list.append(m_patch_bin)
        connected_m_list = np.array(connected_m_list)

        connected_c_list = []
        for contour in contours_c:
            c_patch_bin = np.zeros_like(c_patch, dtype=np.uint8)
            cv2.drawContours(c_patch_bin, [contour], -1, 1, -1)
            connected_c_list.append(c_patch_bin)
        connected_c_list = np.array(connected_c_list)

        overlap_matrix = np.zeros((len(connected_m_list), len(connected_c_list)), dtype=int)
        distance_matrix = compute_distance_matrix(connected_c_list)

        for i, connected_m_ in enumerate(connected_m_list):
            for j, connected_c_ in enumerate(connected_c_list):
                overlap_matrix[i, j] = np.sum(connected_m_ & connected_c_)  # Compute overlap

        m_clusters = assign_contours_to_clusters(overlap_matrix, 2)
        
        graph_data = construct_graph(distance_matrix, connected_c_list, connected_m_list, m_patch)

        if graph_data is not None and graph_data.x.shape[0] >= 5 and 1 in graph_data.edge_label and graph_data.edge_index.shape[1] >= 5:
            # print(os.path.join(graph_dir, os.path.basename(image_path)[:-4]+"_"+str(idx)+".pt"))
            torch.save(graph_data, os.path.join(graph_dir, os.path.basename(image_path)[:-4]+"_"+str(idx)+".pt"))  # 保存到文件


        # if graph_data is not None and graph_data.x.shape[0] >= 5:
        #     # print(len(connected_c_list), graph_data.x.shape, graph_data.edge_index.shape, graph_data.pos.shape)
        #     # print(graph_data.edge_label)

        #     visualize_graph = cv2.cvtColor(c_patch.copy(), cv2.COLOR_GRAY2RGB)
        #     gt_visualize_graph = cv2.cvtColor(m_patch.copy(), cv2.COLOR_GRAY2RGB)

        #     edge_index = graph_data.edge_index.cpu().numpy()  
        #     pos = graph_data.pos  

        #     for i in range(edge_index.shape[1]):  
        #         start_idx, end_idx = edge_index[:, i] 
        #         start_pos = (int(pos[start_idx][0]), int(pos[start_idx][1]))
        #         end_pos = (int(pos[end_idx][0]), int(pos[end_idx][1]))
        #         if graph_data.edge_label[i] == 1:
        #             visualize_graph = cv2.line(visualize_graph, start_pos, end_pos, (0, 0, 255), 2)  # 画蓝色的边
        #         # else:
        #         #     visualize_graph = cv2.line(visualize_graph, start_pos, end_pos, (255, 0, 0), 2)  # 画蓝色的边

        #     for coor in pos:
        #         visualize_graph = cv2.circle(visualize_graph, (int(coor[0]), int(coor[1])), 5, (0, 255, 0), -1)

        #     m_patch_rgb = cv2.cvtColor(m_patch.copy(), cv2.COLOR_GRAY2RGB)
        #     c_patch_rgb = cv2.cvtColor(c_patch.copy(), cv2.COLOR_GRAY2RGB)

        #     red_channel = m_patch.copy()  
        #     green_channel = np.zeros_like(m_patch) 
        #     blue_channel = c_patch.copy()  

        #     combined_image = cv2.merge([blue_channel, green_channel, red_channel])

        #     visualize_graph = np.vstack((visualize_graph, combined_image, c_patch_rgb, m_patch_rgb))
        #     # print(visualize_graph.shape)

        #     cv2.imwrite("./test_visualize/test_visualize_"+str(np.random.randint(1000))+".png", visualize_graph)

        for (m_key, m_partner) in m_clusters.items():
            # print ('key: ', key,'value: ', len(value))
            if len(m_partner)>=2 and np.sum(connected_m_list[m_key]) > 50:
                mapping_block = np.zeros_like(connected_m_list[m_key])
                for c_idx in m_partner:
                    # print(c_idx)
                    mapping_block += connected_c_list[c_idx]
                mapping_block = np.where(mapping_block > 0, 1.0, 0.0).astype(np.uint8)

                rehab_map_path = os.path.join(rehab_dir, f"{os.path.basename(coarse_path).replace('.png', '')}_map_{idx}_{m_key}.png")
                rehab_cluster_path = os.path.join(rehab_dir, f"{os.path.basename(coarse_path).replace('.png', '')}_cluster_{idx}_{m_key}.png")
                roi_path = os.path.join(rehab_dir, f"{os.path.basename(coarse_path).replace('.png', '')}_roi_{idx}_{m_key}.png")
                
                cv2.imwrite(rehab_map_path, mapping_block*255)
                cv2.imwrite(rehab_cluster_path, connected_m_list[m_key]*255)
                cv2.imwrite(roi_path, img_patch)



# Process all images with progress bar
coarse_files = sorted(os.listdir(coarse_dir))
for file in tqdm(coarse_files, desc="Processing Images"):
    coarse_path = os.path.join(coarse_dir, file)
    mask_path = os.path.join(mask_dir, file.replace("coarse", "vessel"))
    image_path = os.path.join(image_dir, file.replace("coarse", "images").replace("png", "tif"))
    process_image(coarse_path, mask_path, image_path, graph_data_dir, rehabilitate_dir)
    
    # break

