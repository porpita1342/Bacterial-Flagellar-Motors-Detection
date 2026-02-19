import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn.functional as F

#All losses and eval metrics are stored here.

__all__ = ['comp_score', 'detect_peaks', 'nms_coords', 'fbeta_score_coords', 'compute_fbeta']

class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def distance_metric(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    thresh_ratio: float,
    min_radius: float,
):
    coordinate_cols = ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']
    label_tensor = solution[coordinate_cols].values.reshape(len(solution), -1, len(coordinate_cols))
    predicted_tensor = submission[coordinate_cols].values.reshape(len(submission), -1, len(coordinate_cols))
    # Find the minimum euclidean distances between the true and predicted points
    solution['distance'] = np.linalg.norm(label_tensor - predicted_tensor, axis=2).min(axis=1)
    # Convert thresholds from angstroms to voxels
    solution['thresholds'] = solution['Voxel spacing'].apply(lambda x: (min_radius * thresh_ratio) / x)
    solution['predictions'] = submission['Has motor'].values
    solution.loc[(solution['distance'] > solution['thresholds']) & (solution['Has motor'] == 1) & (submission['Has motor'] == 1), 'predictions'] = 0
    return solution['predictions'].values


def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, min_radius: float, beta: float) -> float:
    """
    Parameters:
    solution (pd.DataFrame): DataFrame containing ground truth motor positions.
    submission (pd.DataFrame): DataFrame containing predicted motor positions.

    Returns:
    float: FBeta score.

    Example
    --------
    >>> solution = pd.DataFrame({
    ...     'tomo_id': [0, 1, 2, 3],
    ...     'Motor axis 0': [-1, 250, 100, 200],
    ...     'Motor axis 1': [-1, 250, 100, 200],
    ...     'Motor axis 2': [-1, 250, 100, 200],
    ...     'Voxel spacing': [10, 10, 10, 10],
    ...     'Has motor': [0, 1, 1, 1]
    ... })
    >>> submission = pd.DataFrame({
    ...     'tomo_id': [0, 1, 2, 3],
    ...     'Motor axis 0': [100, 251, 600, -1],
    ...     'Motor axis 1': [100, 251, 600, -1],
    ...     'Motor axis 2': [100, 251, 600, -1]
    ... })
    >>> score(solution, submission, 1000, 2)
    0.3571428571428571
    """

    solution = solution.sort_values('tomo_id').reset_index(drop=True)
    submission = submission.sort_values('tomo_id').reset_index(drop=True)

    filename_equiv_array = solution['tomo_id'].eq(submission['tomo_id'], fill_value=0).values

    if np.sum(filename_equiv_array) != len(solution['tomo_id']):
        raise ValueError('Submitted tomo_id values do not match the sample_submission file')

    submission['Has motor'] = 1
    # If any columns are missing an axis, it's marked with no motor
    select = (submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] == -1).any(axis='columns')
    submission.loc[select, 'Has motor'] = 0

    cols = ['Has motor', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']
    assert all(col in submission.columns for col in cols)

    # Calculate a label of 0 or 1 using the 'has motor', and 'motor axis' values
    predictions = distance_metric(
        solution,
        submission,
        thresh_ratio=1.0,
        min_radius=1000.0, #As stated in the competition
    )

    return sklearn.metrics.fbeta_score(solution['Has motor'].values, predictions, beta=beta)


def detect_peaks(fg_prob, threshold=0.3, kernel_size=7):
    """
    Detect local maxima in a 3D foreground probability map using max pooling.

    fg_prob: torch.Tensor of shape (Z, Y, X)
    threshold: minimum confidence to report a peak
    kernel_size: neighbourhood size for local maximum check (odd integer)

    Returns:
        coords: LongTensor of shape (N, 3) — (z, y, x) of each peak
        values: FloatTensor of shape (N,) — confidence at each peak
    """
    prob_5d = fg_prob.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
    pooled = F.max_pool3d(prob_5d, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    is_peak = (prob_5d >= pooled - 1e-6) & (prob_5d > threshold)
    coords = is_peak.squeeze(0).squeeze(0).nonzero(as_tuple=False)  # (N, 3)
    if len(coords) == 0:
        return coords, torch.tensor([], dtype=torch.float32, device=fg_prob.device)
    values = fg_prob[coords[:, 0], coords[:, 1], coords[:, 2]]
    return coords, values


def nms_coords(pred_list, min_dist_voxels):
    """
    Greedy NMS on a list of (z, y, x, confidence) predictions.
    Keeps the highest-confidence detection and suppresses any within
    min_dist_voxels of it.

    Returns: list of (z, y, x) tuples.
    """
    if not pred_list:
        return []
    sorted_preds = sorted(pred_list, key=lambda p: p[3], reverse=True)
    kept = []
    suppressed = [False] * len(sorted_preds)
    for i, (z, y, x, _) in enumerate(sorted_preds):
        if suppressed[i]:
            continue
        kept.append((z, y, x))
        for j in range(i + 1, len(sorted_preds)):
            if suppressed[j]:
                continue
            oz, oy, ox, _ = sorted_preds[j]
            dist = np.sqrt((z - oz)**2 + (y - oy)**2 + (x - ox)**2)
            if dist < min_dist_voxels:
                suppressed[j] = True
    return kept


def fbeta_score_coords(pred_coords, gt_coords, threshold_voxels, beta=2.0):
    """
    Compute F_beta score for coordinate-based motor detection.

    Greedy nearest-neighbour matching: each GT motor is matched to the
    closest unmatched prediction. A match is a TP only if the distance
    is ≤ threshold_voxels (= 1000 Å / voxel_spacing).

    Returns: (tp, fp, fn) counts.
    """
    if not gt_coords and not pred_coords:
        return 0, 0, 0
    if not gt_coords:
        return 0, len(pred_coords), 0
    if not pred_coords:
        return 0, 0, len(gt_coords)

    matched_preds = set()
    tp = 0
    for gt in gt_coords:
        best_dist = float('inf')
        best_idx = -1
        for i, pred in enumerate(pred_coords):
            if i in matched_preds:
                continue
            dist = np.sqrt((gt[0] - pred[0])**2 + (gt[1] - pred[1])**2 + (gt[2] - pred[2])**2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0 and best_dist <= threshold_voxels:
            tp += 1
            matched_preds.add(best_idx)

    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp
    return tp, fp, fn


def compute_fbeta(tp, fp, fn, beta=2.0):
    """F_beta from TP/FP/FN counts. beta=2 weights recall 4x more than precision."""
    denom = (1 + beta**2) * tp + beta**2 * fn + fp
    if denom == 0:
        return 0.0
    return (1 + beta**2) * tp / denom
