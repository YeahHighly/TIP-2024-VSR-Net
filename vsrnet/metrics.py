import numpy as np
import cv2
from medpy.metric import binary
from skimage.morphology import skeletonize
from scipy.stats import wasserstein_distance
from skimage.measure import label, regionprops
from BettiMatching import BettiMatching

def ece_score(py, y_test, n_bins=10):
    """Expected Calibration Error (ECE) computation."""
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = [py[i, py_index[i]] for i in range(py.shape[0])]
    py_value = np.array(py_value)
    
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if a < py_value[i] <= b:
                Bm[m] += 1
                acc[m] += (py_index[i] == y_test[i])
                conf[m] += py_value[i]
                
        if Bm[m] != 0:
            acc[m] /= Bm[m]
            conf[m] /= Bm[m]
    
    ece = sum(Bm[m] * np.abs(acc[m] - conf[m]) for m in range(n_bins)) / sum(Bm)
    return ece

def compute_beta_error(mask_skeleton, pred_skeleton, T):
    """Compute Betti number error for topology-based evaluation."""
    BM = BettiMatching(mask_skeleton, pred_skeleton, relative=False, reduced=False,
                       filtration='superlevel', construction='V', comparison='union')
    BM.get_matching()
    return (BM.Betti_number_error(threshold=0.5, dimensions=[0, 1])) * T[3]

def count_crossings(skeleton):
    """Count crossings in skeleton image."""
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])  
    return (cv2.filter2D(skeleton, -1, kernel) == 13).sum()

def fractal_dimension(skeleton, threshold=0.9):
    """Compute fractal dimension of the vessel structure."""
    skeleton = skeleton > threshold
    sizes = np.logspace(1, np.log2(min(skeleton.shape)), num=10, base=2).astype(int)
    Ns = [(np.add.reduceat(np.add.reduceat(skeleton, np.arange(0, skeleton.shape[0], size), axis=0),
           np.arange(0, skeleton.shape[1], size), axis=1) > 0).sum() for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(Ns), 1)
    return -coeffs[0]

def vessel_tortuosity(skeleton):
    """Compute vessel tortuosity by analyzing skeleton path length."""
    labels = label(skeleton)
    tortuosity_values = []
    for region in regionprops(labels):
        coords = region.coords
        if len(coords) < 2:
            continue
        euclidean_dist = np.linalg.norm(coords[-1] - coords[0])
        path_length = sum(np.linalg.norm(coords[i] - coords[i - 1]) for i in range(1, len(coords)))
        if path_length > 0:
            tortuosity_values.append(path_length / euclidean_dist)
    return np.mean(tortuosity_values) if tortuosity_values else 0

def compute_metrics(mask, pred):
    """
    Compute various segmentation evaluation metrics for a single pair of mask and prediction.
    
    Args:
        mask (np.array): Ground truth binary mask (shape: 256x256).
        pred (np.array): Predicted binary mask (shape: 256x256).
    
    Returns:
        dict: Computed metric values.
    """
    T = (1.0, 0.01, 10, 0.01, 4, 0.1)  # Scaling factors

    pa = np.mean(pred == mask)  
    dice = binary.dc(pred, mask)      
    jaccard = binary.jc(pred, mask)   

    mask_skeleton = skeletonize(mask).astype(np.uint8)
    pred_skeleton = skeletonize(pred).astype(np.uint8)

    mask_crossings = count_crossings(mask_skeleton) 
    pred_crossings = count_crossings(pred_skeleton)
    VBN = abs(mask_crossings - pred_crossings) / (T[0] * (max(mask_crossings, pred_crossings) + 1e-6))

    mask_fractal = fractal_dimension(mask_skeleton)
    pred_fractal = fractal_dimension(pred_skeleton)
    FD = abs(mask_fractal - pred_fractal) / (T[1] * (max(mask_fractal, pred_fractal) + 1e-6))
    
    mask_tortuosity = vessel_tortuosity(mask_skeleton)
    pred_tortuosity = vessel_tortuosity(pred_skeleton)
    VT = abs(mask_tortuosity - pred_tortuosity) / (T[2] * (max(mask_tortuosity, pred_tortuosity) + 1e-6))

    beta_error = compute_beta_error(mask_skeleton, pred_skeleton, T)

    smd_diff = wasserstein_distance(mask_skeleton.reshape(-1), pred_skeleton.reshape(-1)) * T[4]

    ece_score_seg = ece_score(mask_skeleton, pred_skeleton) * T[5]

    return {
        "PA": pa,
        "Dice": dice,
        "Jaccard": jaccard,
        "VBN": VBN,
        "FD": FD,
        "VT": VT,
        "Beta_Error": beta_error,  
        "SMD": smd_diff,
        "ECE": ece_score_seg
    }

def compute_metrics_simple(mask, pred):
    """
    Compute basic segmentation evaluation metrics for a single pair of mask and prediction.
    
    Args:
        mask (np.array): Ground truth binary mask (shape: 256x256).
        pred (np.array): Predicted binary mask (shape: 256x256).
    
    Returns:
        dict: Computed metric values (PA, Dice, Jaccard).
    """
    return {
        "PA": np.mean(pred == mask),
        "Dice": binary.dc(pred, mask),
        "Jaccard": binary.jc(pred, mask)
    }

# # Example Usage:
# # Load images and compute metrics
# mask_path = "mask.png"
# pred_path = "pred.png"

# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
# pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) > 0

# mask = mask.astype(np.uint8)
# pred = pred.astype(np.uint8)

# # Compute the metrics for a single sample
# metrics = compute_metrics(mask, pred)
# print(metrics)
