"""
SEGS composition nodes - reconstruct images from processed segments.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .core import SEG


class XJSegsStitcher:
    """
    Stitch processed segment back into original image.
    Uses feathering for seamless blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "seg": ("SEG",),
                "feather_pixels": (
                    "INT",
                    {"default": 5, "min": 0, "max": 100, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "stitch"

    def stitch(self, original_image, processed_image, seg, feather_pixels):
        """
        Stitch processed image back into original image with feathering.

        Args:
            original_image: Original full image (B, H, W, C)
            processed_image: Processed cropped image (B, H', W', C)
            seg: SEG containing mask and crop region
            feather_pixels: Number of pixels to feather at edges

        Returns:
            Stitched image
        """
        # Extract mask and coordinates from SEG
        mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region

        # Convert mask to torch if needed
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32) / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Ensure mask has batch dimension and correct shape (B, H, W)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        elif len(mask.shape) == 3:
            if mask.shape[2] == 1:
                mask = mask.squeeze(-1).unsqueeze(0)
        elif len(mask.shape) == 4:
            mask = mask.squeeze(-1)

        # Clone original to avoid modifying input
        result = original_image.clone()

        batch_size = result.shape[0]
        orig_h, orig_w = result.shape[1:3]

        # Validate crop region
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(x1, min(x2, orig_w))
        y2 = max(y1, min(y2, orig_h))

        crop_h = y2 - y1
        crop_w = x2 - x1

        if crop_h <= 0 or crop_w <= 0:
            # Invalid crop region, return original
            return (result,)

        # Resize processed image and mask to match crop region
        if processed_image.shape[1] != crop_h or processed_image.shape[2] != crop_w:
            # Resize (B, H, W, C) -> (B, C, H, W) for F.interpolate
            processed_image = processed_image.permute(0, 3, 1, 2)
            processed_image = F.interpolate(
                processed_image,
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=False,
            )
            processed_image = processed_image.permute(
                0, 2, 3, 1
            )  # Back to (B, H, W, C)

        if mask.shape[1] != crop_h or mask.shape[2] != crop_w:
            # Resize mask (B, H, W) -> (B, 1, H, W) for F.interpolate
            mask = mask.unsqueeze(1)
            mask = F.interpolate(
                mask, size=(crop_h, crop_w), mode="bilinear", align_corners=False
            )
            mask = mask.squeeze(1)  # Back to (B, H, W)

        # Apply feathering to mask
        if feather_pixels > 0:
            mask = self._feather_mask(mask, feather_pixels)

        # Expand mask to match image channels (B, H, W) -> (B, H, W, 1)
        blend_mask = mask.unsqueeze(-1)

        # Blend processed image into original
        for b in range(min(batch_size, processed_image.shape[0])):
            # Extract the crop region from original
            original_crop = result[b : b + 1, y1:y2, x1:x2, :]

            # Blend using mask
            blended = processed_image[b : b + 1] * blend_mask[
                b : b + 1
            ] + original_crop * (1 - blend_mask[b : b + 1])

            # Place back into result
            result[b : b + 1, y1:y2, x1:x2, :] = blended

        return (result,)

    def _feather_mask(self, mask, feather_pixels):
        """
        Apply Gaussian blur to mask edges for feathering.

        Args:
            mask: (B, H, W) tensor
            feather_pixels: radius of feathering

        Returns:
            Feathered mask (B, H, W)
        """
        # Convert to (B, 1, H, W) for conv2d
        mask = mask.unsqueeze(1)

        # Create Gaussian kernel
        kernel_size = feather_pixels * 2 + 1
        sigma = feather_pixels / 3.0  # Standard sigma for Gaussian

        # Create 1D Gaussian kernel
        x = (
            torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            - feather_pixels
        )
        gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()

        # Create 2D kernel (outer product)
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

        # Apply Gaussian blur with padding
        padding = feather_pixels
        blurred = F.conv2d(mask, kernel, padding=padding)

        # Remove channel dimension
        blurred = blurred.squeeze(1)

        # Clamp to [0, 1]
        blurred = torch.clamp(blurred, 0.0, 1.0)

        return blurred


class XJSegsMerge:
    """
    Merge multiple SEGs into a single merged SEG.
    - bboxes: covers all segments
    - confidence: maximum confidence
    - label: from the most confident segment
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "merge"

    def merge(self, segs):
        """
        Merge all SEGs into a single SEG.

        Args:
            segs: SEGS tuple ((height, width), [SEG, ...])

        Returns:
            SEGS with a single merged SEG
        """
        shape = segs[0]
        segments = segs[1]

        if not segments:
            # Empty SEGS, return as-is
            return (segs,)

        # Initialize bounding box tracking
        crop_left = sys.maxsize
        crop_right = 0
        crop_top = sys.maxsize
        crop_bottom = 0

        bbox_left = sys.maxsize
        bbox_right = 0
        bbox_top = sys.maxsize
        bbox_bottom = 0

        # Track max confidence and corresponding label
        max_confidence = -1.0
        max_confidence_label = "merged"

        # Find bounding boxes that cover all segments and max confidence
        for seg in segments:
            cx1, cy1, cx2, cy2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            crop_left = min(crop_left, cx1)
            crop_top = min(crop_top, cy1)
            crop_right = max(crop_right, cx2)
            crop_bottom = max(crop_bottom, cy2)

            bbox_left = min(bbox_left, bx1)
            bbox_top = min(bbox_top, by1)
            bbox_right = max(bbox_right, bx2)
            bbox_bottom = max(bbox_bottom, by2)

            # Track max confidence and its label
            if seg.confidence > max_confidence:
                max_confidence = seg.confidence
                max_confidence_label = seg.label

        # If no segments had valid confidence, use 0.0
        if max_confidence < 0:
            max_confidence = 0.0

        # Combine all masks using OR operation
        combined_mask = self._combine_masks(segs)

        # Crop the combined mask to the merged crop region
        cropped_mask = combined_mask[crop_top:crop_bottom, crop_left:crop_right]
        cropped_mask = cropped_mask.unsqueeze(0)

        crop_region = [crop_left, crop_top, crop_right, crop_bottom]
        bbox = [bbox_left, bbox_top, bbox_right, bbox_bottom]

        # Create merged SEG with max confidence and label from most confident segment
        merged_seg = SEG(
            None, cropped_mask, max_confidence, crop_region, bbox, max_confidence_label, None
        )

        return ((shape, [merged_seg]),)

    def _combine_masks(self, segs):
        """
        Combine all segment masks using OR operation.

        Args:
            segs: SEGS tuple

        Returns:
            Combined mask tensor (H, W)
        """
        shape = segs[0]
        h, w = shape[0], shape[1]

        mask = np.zeros((h, w), dtype=np.uint8)

        for seg in segs[1]:
            cropped_mask = seg.cropped_mask
            crop_region = seg.crop_region

            # Handle both numpy arrays and tensors
            if isinstance(cropped_mask, np.ndarray):
                mask_data = (cropped_mask * 255).astype(np.uint8)
            else:
                # Convert tensor to numpy
                mask_data = (cropped_mask * 255).cpu().numpy().astype(np.uint8)

            # Handle 3D masks by squeezing to 2D
            if mask_data.ndim == 3:
                mask_data = mask_data.squeeze(0)

            # OR operation to combine masks
            mask[crop_region[1] : crop_region[3], crop_region[0] : crop_region[2]] |= (
                mask_data
            )

        return torch.from_numpy(mask.astype(np.float32) / 255.0)


NODE_CLASS_MAPPINGS = {
    "XJSegsStitcher": XJSegsStitcher,
    "XJSegsMerge": XJSegsMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSegsStitcher": "SEGS Stitcher",
    "XJSegsMerge": "SEGS Merge",
}
