"""
Core SEGS data structures and utilities.
"""

from collections import namedtuple

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


def get_segs_list(segs):
    """Get the list of SEG elements from SEGS."""
    return segs[1]


def count_segs(segs):
    """Count the number of segments in SEGS."""
    return len(segs[1])
