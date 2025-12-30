"""
SEGS decomposition nodes - break SEGS into standard ComfyUI types.
"""

import numpy as np
import torch
from PIL import Image


class XJSegsExtractor:
    """
    Extract components from a single SEG into standard ComfyUI types.
    Decomposes SEG into IMAGE, MASK, and metadata for use in standard workflows.

    When image is provided, crops from that image using crop_region.
    This enables iterative workflows where later segments see improvements from earlier segments.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seg": ("SEG",),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "mask", "x1", "y1", "x2", "y2", "label", "confidence")
    CATEGORY = "XJNodes/segs"
    FUNCTION = "extract"

    def extract(self, seg, image=None):
        """
        Extract SEG components.

        Args:
            seg: SEG to extract from
            image: Optional current image to crop from (enables iterative workflows)

        Returns:
            image: cropped image (IMAGE)
            mask: cropped mask (MASK)
            x1, y1, x2, y2: crop region coordinates (for stitcher)
            label: segment label
            confidence: detection confidence
        """
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        label = seg.label if seg.label else ""
        confidence = seg.confidence

        # Crop from provided image if available, otherwise use seg.cropped_image
        if image is not None:
            cropped_image = self._crop_image(image, x1, y1, x2, y2)
        else:
            cropped_image = seg.cropped_image

        # Convert mask: PIL -> numpy -> torch
        if isinstance(cropped_mask, Image.Image):
            cropped_mask = np.array(cropped_mask).astype(np.float32) / 255.0
        if isinstance(cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(cropped_mask)

        # Convert image: PIL -> numpy -> torch
        if cropped_image is not None:
            if isinstance(cropped_image, Image.Image):
                cropped_image = np.array(cropped_image).astype(np.float32) / 255.0
            if isinstance(cropped_image, np.ndarray):
                cropped_image = torch.from_numpy(cropped_image)

        # Ensure image is in correct format (B, H, W, C)
        if cropped_image is not None:
            if len(cropped_image.shape) == 2:
                # Grayscale (H, W) -> (H, W, 1) -> (1, H, W, 1)
                cropped_image = cropped_image.unsqueeze(-1).unsqueeze(0)
            elif len(cropped_image.shape) == 3:
                # (H, W, C) -> (1, H, W, C)
                cropped_image = cropped_image.unsqueeze(0)
        else:
            # If no image, create empty tensor matching mask shape
            if len(cropped_mask.shape) == 2:
                h, w = cropped_mask.shape
                cropped_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
            else:
                h, w = cropped_mask.shape[1:3]
                cropped_image = torch.zeros((1, h, w, 3), dtype=torch.float32)

        # Ensure mask is in correct format (B, H, W)
        if len(cropped_mask.shape) == 2:
            # Add batch dimension if missing (H, W) -> (1, H, W)
            cropped_mask = cropped_mask.unsqueeze(0)
        elif len(cropped_mask.shape) == 3:
            # Could be (H, W, 1) or (B, H, W)
            # Check if last dimension is 1 (channel dimension)
            if cropped_mask.shape[2] == 1:
                # (H, W, 1) -> (H, W) -> (1, H, W)
                cropped_mask = cropped_mask.squeeze(-1).unsqueeze(0)
            # else assume it's already (B, H, W)
        elif len(cropped_mask.shape) == 4:
            # (B, H, W, 1) -> (B, H, W)
            cropped_mask = cropped_mask.squeeze(-1)

        return (cropped_image, cropped_mask, x1, y1, x2, y2, label, confidence)

    def _crop_image(self, image, x1, y1, x2, y2):
        """
        Crop image using crop_region coordinates.

        Args:
            image: Full image (B, H, W, C)
            x1, y1, x2, y2: Crop region coordinates

        Returns:
            Cropped image (B, H', W', C)
        """
        # Validate and clamp coordinates
        batch_size, orig_h, orig_w, channels = image.shape
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(x1, min(x2, orig_w))
        y2 = max(y1, min(y2, orig_h))

        # Crop the image
        cropped = image[:, y1:y2, x1:x2, :]

        return cropped


NODE_CLASS_MAPPINGS = {
    "XJSegsExtractor": XJSegsExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSegsExtractor": "SEGS Extractor",
}
