import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import argparse
from torchvision.utils import save_image
import module
import dataloader
import cv2
from torch_geometric.data import Data
from glob import glob
from scipy.spatial.distance import cdist
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from metrics import compute_metrics, compute_metrics_simple
from prettytable import PrettyTable
from medpy.metric import binary
import random
random.seed(1024)
np.random.seed(1024)

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

    return np.array(centroids) if centroids else np.empty((0, 2)), np.array(center_point) if center_point else np.empty((0, 2))  

def construct_graph(distance_matrix, connected_c_list, connected_m_list, m_patch, threshold=100, top_k=20):
    coordinates, center_points = get_centroids(connected_c_list)

    # Extract node features using ResNet18
    connected_list_tensor = [transform_ccm(cv2.cvtColor(connected, cv2.COLOR_GRAY2RGB)) for connected in connected_c_list]
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

def compute_distance_matrix(connected_c_list):

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

# ------------------------ Patch Extraction ------------------------
def extract_patches(image, patch_size=256, grid_size=4):

    h, w = image.shape[:2]
    patch_h, patch_w = h // grid_size, w // grid_size
    patches = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            patch = image[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
            patch = cv2.resize(patch, (patch_size, patch_size))
            patches.append(patch)
    
    return patches

def find(parent, x):
    """Find the root of node x with path compression."""
    if parent[x] != x:
        parent[x] = find(parent, parent[x])  # Path compression
    return parent[x]

def union(parent, rank, x, y):
    """Union by rank: merge two sets containing x and y."""
    root_x = find(parent, x)
    root_y = find(parent, y)
    
    if root_x != root_y:
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

def get_clusters(graph_data, edge_preds):

    edge_index = graph_data.edge_index  # Shape: [2, num_edges]

    if edge_preds.ndim == 0:
        edge_preds = np.array([edge_preds])
    
    # Ensure edge_preds is a NumPy array or tensor to allow correct indexing
    edge_preds = torch.tensor(edge_preds).cpu().numpy() if isinstance(edge_preds, list) else edge_preds
    num_nodes = graph_data.num_nodes

    # Initialize the Union-Find structure
    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    # Merge nodes that are connected with edge_label = 1
    for i in range(edge_index.shape[1]):
        if edge_preds[i] == 1:  # Ensure correct indexing
            node_u = edge_index[0, i].item()
            node_v = edge_index[1, i].item()
            union(parent, rank, node_u, node_v)

    # Group nodes into clusters
    clusters = {}
    for node in range(num_nodes):
        root = find(parent, node)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(node)

    return list(clusters.values())

# ------------------------ Argument Parser ------------------------
parser = argparse.ArgumentParser(description="Graph Edge Classification - Inference")
parser.add_argument("--ccm_module", type=str, default='ccm_plus', help="Graph classification backbone, 'ccm' and 'ccm_plus'")
parser.add_argument("--cmm_module", type=str, default='cmm', help="Graph classification backbone, 'cmm' and 'cmm_plus'")
parser.add_argument("--dataset", type=str, default='drive', help="Graph dataset.")
parser.add_argument("--model_path", type=str, default="./checkpoints/", help="Path to saved model checkpoints.")
args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ Directory Setup ------------------------
if args.dataset.lower() == 'drive':
    base_path = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/DRIVE_AV"
    # Define directories
    coarse_dir       = os.path.join(base_path, "testing", "coarse")
    mask_dir         = os.path.join(base_path, "testing", "vessel")
    image_dir        = os.path.join(base_path, "testing", "images")
    results_dir   = os.path.join(base_path, "testing", "inference")
elif args.dataset.lower() == 'octa':
    base_path = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/OCTA-500"
    coarse_dir       = os.path.join(base_path, "coarse")
    mask_dir         = os.path.join(base_path, "OCTAFULL")
    image_dir        = os.path.join(base_path, "OCTAFULL")
    graph_data_dir   = os.path.join(base_path, "graph_data")
    rehabilitate_dir = os.path.join(base_path, "rehabilitate")
else:
    raise ValueError("Unsupported dataset. Use 'drive' or 'octa'.")

os.makedirs(results_dir, exist_ok=True)

ccm_factory = module.CCMFactory()
ccm_model = ccm_factory.get_model(args.ccm_module, num_classes=1)
ccm_model.load_state_dict(torch.load(os.path.join(args.model_path, args.dataset, args.ccm_module, "best.pth"), map_location=DEVICE, weights_only=True), strict=False)
ccm_model.to(DEVICE)
ccm_model.eval()

cmm_factory = module.CMMFactory()
cmm_model = cmm_factory.get_model(args.cmm_module)
cmm_model.load_state_dict(torch.load(os.path.join(args.model_path, args.dataset, args.cmm_module, "best.pth"), map_location=DEVICE, weights_only=True), strict=False)
cmm_model.to(DEVICE)
cmm_model.eval()

patch_size = 256

# Image transformation
transform_ccm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((patch_size, patch_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_cmm = A.Compose([
                A.Resize(patch_size, patch_size),
                ToTensorV2()
            ])

# ------------------------ Process Images ------------------------
if args.dataset.lower() == 'drive':
    image_files = sorted(glob(os.path.join(image_dir, "*.tif")))
else:
    image_files = sorted(glob(os.path.join(image_dir, "*.bmp")))
    random.shuffle(image_files)
    image_files = image_files[int(len(image_files)*0.8):]

threshold_ccm = 0.3

for img_file in image_files:
    image = cv2.imread(img_file, cv2.COLOR_BGR2RGB)
    coarse = cv2.imread(os.path.join(coarse_dir, os.path.basename(img_file).replace('.tif', '.png')), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(mask_dir, os.path.basename(img_file).replace('.tif', '.png')), cv2.IMREAD_GRAYSCALE)
    
    image_patches = extract_patches(image)
    coarse_patches = extract_patches(coarse)
    mask_patches = extract_patches(mask)

    total_preds_mapping = np.zeros((4*patch_size, 4*patch_size))
    
    for idx, (img_patch, coarse_patch, mask_patch) in enumerate(zip(image_patches, coarse_patches, mask_patches)):
        kernel = np.ones((3,3),np.uint8)  
        c_patch = cv2.morphologyEx(coarse_patch, cv2.MORPH_CLOSE, kernel)
        m_patch = cv2.morphologyEx(mask_patch, cv2.MORPH_CLOSE, kernel)

        c_patch = cv2.dilate(c_patch, kernel, iterations = 1)
        m_patch = cv2.dilate(m_patch, kernel, iterations = 1)

        c_patch = cv2.erode(c_patch, kernel, iterations = 1)
        m_patch = cv2.erode(m_patch, kernel, iterations = 1)
        
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
        
        graph_data = construct_graph(distance_matrix, connected_c_list, connected_m_list, m_patch)

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

        #     cv2.imwrite("./test_visualize_total/test_visualize_"+str(np.random.randint(1000))+".png", visualize_graph)
        if graph_data is not None:
            graph_data = graph_data.to(DEVICE)
            outputs = ccm_model(graph_data)
            preds = (outputs >= threshold_ccm).int().cpu().numpy()  # Convert probabilities to binary predictions.cpu().numpy()
            gt = graph_data.edge_label.cpu().numpy()

            cluster_results = get_clusters(graph_data, preds)
            
            preds_mapping = np.zeros_like(c_patch).astype(np.int32)
            for cluster in cluster_results:
                mapping_c = np.zeros_like(c_patch)
                for c_idx in cluster:
                    mapping_c += connected_c_list[c_idx]
                mapping_c = np.where(mapping_c >= 1, 255.0, 0.0)
                image_roi = img_patch
                transformed = transform_cmm(image=image_roi, mask=mapping_c)
                image_roi_tensor = transformed["image"].unsqueeze(0).float()
                mapping_tensor = transformed["mask"].unsqueeze(0).unsqueeze(0).float() / 255.0
                image_roi_tensor = image_roi_tensor.to(DEVICE)
                mapping_tensor = mapping_tensor.to(DEVICE)
                cmm_output = cmm_model(image_roi_tensor, mapping_tensor)
                cmm_output = (cmm_output > 0.5).squeeze().int().cpu().numpy()

                preds_mapping += cmm_output

            transformed = transform_cmm(image=image_roi, mask=coarse_patch)
            image_roi_tensor = transformed["image"].unsqueeze(0).float()
            mapping_tensor = transformed["mask"].unsqueeze(0).unsqueeze(0).float() / 255.0
            image_roi_tensor = image_roi_tensor.to(DEVICE)
            mapping_tensor = mapping_tensor.to(DEVICE)
            cmm_output = cmm_model(image_roi_tensor, mapping_tensor)
            cmm_output = (cmm_output > 0.5).squeeze().int().cpu().numpy()
            preds_mapping += cmm_output

            preds_mapping = np.where(preds_mapping > 0, 255.0, 0.0)
            total_preds_mapping[(idx//4)*patch_size:(idx//4 + 1)*patch_size, (idx%4)*patch_size:(idx%4 + 1)*patch_size] = preds_mapping
        else:
            total_preds_mapping[(idx//4)*patch_size:(idx//4 + 1)*patch_size, (idx%4)*patch_size:(idx%4 + 1)*patch_size] = coarse_patch

    total_preds_mapping = np.where(total_preds_mapping > 0, 255.0, 0.0)
    total_preds_mapping = cv2.resize(total_preds_mapping, (mask.shape[1], mask.shape[0]), cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(results_dir, os.path.basename(img_file).replace('.tif', '.png')), total_preds_mapping)

# all_metrics = {
#     "PA": [],
#     "Dice": [],
#     "Jaccard": [],
#     "VBN": [],
#     "FD": [],
#     "VT": [],
#     "Beta_Error": [],
#     "SMD": [],
#     "ECE": []
# }

all_metrics = {
    "PA": [],
    "Dice": [],
    "Jaccard": [],
}

result_files = sorted(glob(os.path.join(results_dir, "*.png")))

# Note: The original morphological evaluation difference metrics were calculated using MATLAB. 
# The Python output results are for reference only.

for result_name in result_files:
    result_path = os.path.join(results_dir, result_name)
    pred = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(result_path.replace('inference', 'vessel'), cv2.IMREAD_GRAYSCALE)

    mask = (mask > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    # Compute metrics for this batch
    metric_score = compute_metrics_simple(pred, mask)
    # batch_metrics = compute_metrics(preds, targets)

    # Append results to the lists
    for metric_name in all_metrics:
        all_metrics[metric_name].append(metric_score[metric_name])

    # break
# Compute the average for each metric
avg_metrics = {metric_name: np.mean(values) for metric_name, values in all_metrics.items()}

# Display results in a horizontal table format
table = PrettyTable()
table.field_names = ["Metric"] + list(avg_metrics.keys())  # First row as metric names
table.add_row(["Value"] + [f"{avg_metrics[metric]:.4f}" for metric in avg_metrics])  # Second row as values

print(table)