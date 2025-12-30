"""
Core SEGS data structures and utilities.
"""

from collections import namedtuple
import numpy as np
import torch

# SEG: Individual segment element
SEG = namedtuple(
    "SEG",
    [
        "cropped_image",  # Tensor: cropped region from original image (can be None)
        "cropped_mask",  # Tensor: segment mask (0.0-1.0)
        "confidence",  # float: detection confidence
        "crop_region",  # tuple: (x1, y1, x2, y2) crop coordinates in original image space
        "bbox",  # tuple: (x1, y1, x2, y2) detected bounds in cropped image space
        "label",  # str: segment class/label
        "control_net_wrapper",  # optional: ControlNet data
    ],
    defaults=[None],
)

# SEGS: Collection of segments with shape metadata
# Format: ((height, width), [SEG, SEG, ...])


def create_segs(shape, segments):
    """
    Create a SEGS tuple.

    Args:
        shape: tuple (height, width) - original image dimensions
        segments: list of SEG elements

    Returns:
        SEGS tuple: (shape, segments)
    """
    return (shape, segments)


def get_segs_shape(segs):
    """Get the shape (height, width) from SEGS."""
    return segs[0]


def is_valid_seg(seg):
    """
    Check if a SEG is valid (has non-empty mask and valid bbox).

    Invalid SEGs have:
    - Empty masks: tensor/array with 0 elements
    - Invalid bbox: coordinates where start >= end
    """
    # Check if mask is empty
    if seg.cropped_mask is None:
        return False

    # Handle different mask types
    if isinstance(seg.cropped_mask, torch.Tensor):
        if seg.cropped_mask.numel() == 0:
            return False
    elif isinstance(seg.cropped_mask, np.ndarray):
        if seg.cropped_mask.size == 0:
            return False
    else:
        # PIL Image or other types - check if it has size attribute
        try:
            if hasattr(seg.cropped_mask, "size"):
                # PIL Image size is (width, height)
                width, height = seg.cropped_mask.size
                if width == 0 or height == 0:
                    return False
            else:
                # Unknown type, consider invalid
                return False
        except:
            return False

    # Check bbox validity: x1 < x2 and y1 < y2
    if seg.bbox is not None and len(seg.bbox) >= 4:
        x1, y1, x2, y2 = seg.bbox[:4]
        if x1 >= x2 or y1 >= y2:
            return False

    return True


def get_segs_list(segs):
    """Get the list of valid SEG elements from SEGS (filters out invalid/empty SEGs)."""
    return [seg for seg in segs[1] if is_valid_seg(seg)]


def count_segs(segs):
    """Count the number of valid segments in SEGS (excludes invalid/empty SEGs)."""
    return len(get_segs_list(segs))
