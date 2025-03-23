import numpy as np
import cv2
from medpy.metric import binary
from skimage.morphology import skeletonize
from scipy.stats import wasserstein_distance
from skimage.measure import label, regionprops
from BettiMatching import BettiMatching
from sklearn.calibration import calibration_curve


def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def compute_beta_error(mask_skeleton, pred_skeleton, T):
    beta_errors = []
    BM = BettiMatching(mask_skeleton, pred_skeleton, relative=False, reduced=False,
                       filtration='superlevel', construction='V', comparison='union')
    BM.get_matching()
    beta_error = (BM.Betti_number_error(threshold=0.5, dimensions=[0, 1])) * T[3]
    return beta_error

def count_crossings(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])  
    crossings = (cv2.filter2D(skeleton, -1, kernel) == 13).sum()
    return crossings

def fractal_dimension(skeleton, threshold=0.9):
    skeleton = skeleton > threshold
    sizes = np.logspace(1, np.log2(min(skeleton.shape)), num=10, base=2).astype(int)
    Ns = []
    for size in sizes:
        grid = np.add.reduceat(
            np.add.reduceat(skeleton, np.arange(0, skeleton.shape[0], size), axis=0),
            np.arange(0, skeleton.shape[1], size), axis=1)
        Ns.append((grid > 0).sum())
    coeffs = np.polyfit(np.log(sizes), np.log(Ns), 1)
    return -coeffs[0]

def vessel_tortuosity(skeleton):
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

def compute_metrics(mask_batch, pred_batch):

    batch_size = mask_batch.shape[0]
    T = (1.0, 0.01, 10, 0.01, 4, 0.1)  # Some scaling factors for different metrics

    results = {
        "PA": [],
        "Dice": [],
        "Jaccard": [],
        "VBN": [],
        "FD": [],
        "VT": [],
        "Beta_Error": [],
        "SMD": [],
        "ECE": []
    }

    for i in range(batch_size):
        mask = mask_batch[i]
        pred = pred_batch[i]

        pa = np.mean(pred == mask) 
        dice = binary.dc(pred, mask)      
        jaccard = binary.jc(pred, mask)   

        mask_skeleton = skeletonize(mask)
        pred_skeleton = skeletonize(pred)

        mask_skeleton = (mask_skeleton * 1).astype(np.uint8)
        pred_skeleton = (pred_skeleton * 1).astype(np.uint8)

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

        results["PA"].append(pa)
        results["Dice"].append(dice)
        results["Jaccard"].append(jaccard)
        results["VBN"].append(VBN)
        results["FD"].append(FD)
        results["VT"].append(VT)
        results["Beta_Error"].append(beta_error)  
        results["SMD"].append(smd_diff)
        results["ECE"].append(ece_score_seg)

    avg_results = {key: np.mean(value) for key, value in results.items()}
    return avg_results

def compute_metrics_simple(mask_batch, pred_batch):

    batch_size = mask_batch.shape[0]

    results = {
        "PA": [],
        "Dice": [],
        "Jaccard": [],
    }

    for i in range(batch_size):
        mask = mask_batch[i]
        pred = pred_batch[i]

        pa = np.mean(pred == mask) 
        dice = binary.dc(pred, mask)      
        jaccard = binary.jc(pred, mask)   


        results["PA"].append(pa)
        results["Dice"].append(dice)
        results["Jaccard"].append(jaccard)

    avg_results = {key: np.mean(value) for key, value in results.items()}
    return avg_results



# # Load images and expand dimensions for batch processing
# mask_path = "mask.png"
# pred_path = "pred.png"

# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
# pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) > 0

# mask = mask.astype(np.uint8)
# pred = pred.astype(np.uint8)

# # Expand dimensions to simulate a batch (batch_size, h, w)
# mask_batch = np.expand_dims(mask, axis=0)  
# pred_batch = np.expand_dims(pred, axis=0)  

# # Compute the metrics for the batch
# metrics = compute_metrics(mask_batch, pred_batch)
# print(metrics)
