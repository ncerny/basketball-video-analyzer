"""Pure PyTorch fallback for cv_utils kernel functions.

The cv_utils kernel (kernels-community/cv_utils) provides CUDA-only
implementations of NMS and connected components. On MPS (Apple Silicon),
these functions fail to load.

This module provides pure PyTorch implementations that work on any device,
and patches the SAM3 model to use them when cv_utils is unavailable.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def generic_nms(
    ious: torch.Tensor,
    probs: torch.Tensor,
    iou_threshold: float,
    use_iou_matrix: bool = True,
) -> torch.Tensor:
    """Pure PyTorch implementation of greedy NMS using precomputed IoU matrix.

    This is a fallback for cv_utils_kernel.generic_nms when the CUDA kernel
    is not available (e.g., on MPS devices).

    Args:
        ious: Precomputed IoU matrix of shape (N, N) where N is number of detections.
        probs: Confidence scores of shape (N,).
        iou_threshold: IoU threshold for suppression.
        use_iou_matrix: Must be True (we only support precomputed IoUs).

    Returns:
        Tensor of indices of kept detections.
    """
    if not use_iou_matrix:
        raise ValueError("Fallback NMS only supports use_iou_matrix=True")

    device = probs.device
    n = probs.shape[0]

    if n == 0:
        return torch.tensor([], dtype=torch.int64, device=device)

    # Sort by confidence (descending)
    order = torch.argsort(probs, descending=True)

    keep = []
    suppressed = torch.zeros(n, dtype=torch.bool, device=device)

    for i in range(n):
        idx = order[i].item()
        if suppressed[idx]:
            continue

        keep.append(idx)

        # Suppress all detections with IoU > threshold
        for j in range(i + 1, n):
            other_idx = order[j].item()
            if not suppressed[other_idx] and ious[idx, other_idx] > iou_threshold:
                suppressed[other_idx] = True

    return torch.tensor(keep, dtype=torch.int64, device=device)


def cc_2d(
    mask: torch.Tensor,
    get_counts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch implementation of 2D connected components.

    This is a fallback for cv_utils_kernel.cc_2d when the CUDA kernel
    is not available. Uses iterative flood-fill approach.

    Args:
        mask: Binary mask tensor of shape (B, C, H, W) with uint8 dtype.
        get_counts: If True, also return component areas.

    Returns:
        labels: Component labels tensor of shape (B, C, H, W) with int32 dtype.
        counts: Area of each component at each pixel (same shape).
    """
    B, C, H, W = mask.shape
    device = mask.device

    labels = torch.zeros((B, C, H, W), dtype=torch.int32, device=device)
    counts = torch.zeros((B, C, H, W), dtype=torch.int32, device=device)

    # Process each batch and channel
    for b in range(B):
        for c in range(C):
            mask_2d = mask[b, c]
            labels_2d, counts_2d = _cc_2d_single(mask_2d, device)
            labels[b, c] = labels_2d
            counts[b, c] = counts_2d

    return labels, counts


def _cc_2d_single(mask: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Connected components for a single 2D mask using union-find."""
    H, W = mask.shape
    labels = torch.zeros((H, W), dtype=torch.int32, device=device)
    counts = torch.zeros((H, W), dtype=torch.int32, device=device)

    # Simple iterative labeling (not optimized but works on any device)
    current_label = 0
    label_counts = {}

    # First pass: assign initial labels and track equivalences
    parent = {}  # Union-find parent

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Move to CPU for iteration (MPS doesn't support item access in loops well)
    mask_cpu = mask.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    for i in range(H):
        for j in range(W):
            if mask_cpu[i, j] == 0:
                continue

            neighbors = []
            if i > 0 and mask_cpu[i - 1, j] > 0:
                neighbors.append(labels_cpu[i - 1, j])
            if j > 0 and mask_cpu[i, j - 1] > 0:
                neighbors.append(labels_cpu[i, j - 1])

            if not neighbors:
                current_label += 1
                labels_cpu[i, j] = current_label
                parent[current_label] = current_label
            else:
                min_label = min(neighbors)
                labels_cpu[i, j] = min_label
                for n in neighbors:
                    if n != min_label:
                        union(n, min_label)

    # Second pass: resolve equivalences and count
    label_map = {}
    final_label = 0

    for i in range(H):
        for j in range(W):
            if labels_cpu[i, j] > 0:
                root = find(labels_cpu[i, j])
                if root not in label_map:
                    final_label += 1
                    label_map[root] = final_label
                labels_cpu[i, j] = label_map[root]

    # Count pixels per label
    for label in range(1, final_label + 1):
        label_counts[label] = (labels_cpu == label).sum()

    # Fill counts
    counts_cpu = labels.cpu().numpy()
    for i in range(H):
        for j in range(W):
            lbl = labels_cpu[i, j]
            if lbl > 0:
                counts_cpu[i, j] = label_counts.get(lbl, 0)

    return torch.tensor(labels_cpu, dtype=torch.int32, device=device), torch.tensor(
        counts_cpu, dtype=torch.int32, device=device
    )


class CvUtilsFallback:
    """Fallback cv_utils kernel implementation for non-CUDA devices."""

    def __init__(self):
        self.generic_nms = generic_nms
        self.cc_2d = cc_2d


def patch_sam3_cv_utils() -> bool:
    """Patch SAM3 to use fallback cv_utils when CUDA kernel is unavailable.

    This patches the cv_utils_kernel global in the SAM3 video model to use
    our pure PyTorch implementations.

    Returns:
        True if patch was applied, False if cv_utils was already working.
    """
    try:
        from transformers.models.sam3_video import modeling_sam3_video

        # First, try to load the actual cv_utils kernel
        modeling_sam3_video._load_cv_utils_kernel_once()

        if modeling_sam3_video.cv_utils_kernel:
            logger.info("cv_utils kernel loaded successfully, no patch needed")
            return False

        # cv_utils failed, apply our fallback
        logger.info("cv_utils kernel not available, applying PyTorch fallback")
        modeling_sam3_video.cv_utils_kernel = CvUtilsFallback()
        return True

    except ImportError as e:
        logger.warning(f"Could not patch SAM3 cv_utils: {e}")
        return False
