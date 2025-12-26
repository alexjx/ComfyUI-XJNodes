"""
SEGS composition nodes - reconstruct images from processed segments.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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
            "optional": {
                "check_ratio": (
                    "BOOLEAN",
                    {"default": True, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "stitch"

    def stitch(self, original_image, processed_image, seg, feather_pixels, check_ratio=True):
        """
        Stitch processed image back into original image with feathering.

        Args:
            original_image: Original full image (B, H, W, C)
            processed_image: Processed cropped image (B, H', W', C)
            seg: SEG containing mask and crop region
            feather_pixels: Number of pixels to feather at edges
            check_ratio: Validate aspect ratio before stitching

        Returns:
            Stitched image
        """
        # Extract mask and coordinates from SEG
        mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region

        # Calculate expected dimensions from crop_region
        expected_h = y2 - y1
        expected_w = x2 - x1

        # Validate aspect ratio if dimensions changed
        if check_ratio and expected_h > 0 and expected_w > 0:
            actual_h, actual_w = processed_image.shape[1:3]

            if (actual_h != expected_h or actual_w != expected_w):
                expected_ratio = expected_w / expected_h
                actual_ratio = actual_w / actual_h
                tolerance = 0.01  # 1% tolerance for floating point

                if abs(expected_ratio - actual_ratio) > tolerance:
                    raise ValueError(
                        f"Aspect ratio mismatch: "
                        f"expected {expected_ratio:.3f} (from crop_region {expected_w}x{expected_h}), "
                        f"got {actual_ratio:.3f} (from processed image {actual_w}x{actual_h}). "
                        f"Cannot safely resize - the image was likely cropped or incorrectly resized."
                    )

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


NODE_CLASS_MAPPINGS = {
    "XJSegsStitcher": XJSegsStitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSegsStitcher": "SEGS Stitcher",
}
