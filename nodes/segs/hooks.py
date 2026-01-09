"""
SEGS hook providers for Impact Pack detailer integration.
"""

import torch
from torch import Tensor
import numpy as np


def calc_mean_std(feat: Tensor, eps=1e-5):
    """
    Calculate mean and std for adaptive instance normalization.

    Args:
        feat: 4D tensor (B, C, H, W)
        eps: Small value to avoid divide-by-zero

    Returns:
        mean, std tensors
    """
    size = feat.size()
    assert len(size) == 4, "Input must be 4D tensor"
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor):
    """
    Adaptive instance normalization for color transfer.
    Matches mean and std statistics of content to style.

    Args:
        content_feat: Target image tensor (B, C, H, W)
        style_feat: Reference image tensor (B, C, H, W)

    Returns:
        Color-matched tensor
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adain_color_match(target: Tensor, reference: Tensor):
    """
    Apply AdaIN color matching.

    Args:
        target: Target image (B, H, W, C) in [0, 1]
        reference: Reference image (B, H, W, C) in [0, 1]

    Returns:
        Color-matched target (B, H, W, C)
    """
    # Convert to (B, C, H, W) for AdaIN
    target_t = target.permute(0, 3, 1, 2)
    reference_t = reference.permute(0, 3, 1, 2)

    # Apply AdaIN
    result_t = adaptive_instance_normalization(target_t, reference_t)

    # Convert back to (B, H, W, C)
    result = result_t.permute(0, 2, 3, 1)

    return result.clamp(0.0, 1.0)


def color_matcher_transfer(target: Tensor, reference: Tensor, method: str):
    """
    Apply color matching using color-matcher library.

    Args:
        target: Target image (B, H, W, C) in [0, 1]
        reference: Reference image (B, H, W, C) in [0, 1]
        method: One of mkl, hm, reinhard, mvgd, hm-mvgd-hm, hm-mkl-hm

    Returns:
        Color-matched target (B, H, W, C)
    """
    try:
        from color_matcher import ColorMatcher
    except ImportError:
        raise ImportError(
            "color-matcher library required for methods: mkl, hm, reinhard, mvgd, "
            "hm-mvgd-hm, hm-mkl-hm. Install with: pip install color-matcher"
        )

    target_cpu = target.cpu()
    reference_cpu = reference.cpu()
    batch_size = target_cpu.size(0)

    out = []
    cm = ColorMatcher()

    for i in range(batch_size):
        target_np = target_cpu[i].numpy()
        reference_np = reference_cpu[i].numpy() if reference_cpu.size(0) > 1 else reference_cpu[0].numpy()

        try:
            matched_np = cm.transfer(src=target_np, ref=reference_np, method=method)
            out.append(torch.from_numpy(matched_np))
        except Exception as e:
            print(f"[XJSegsColorMatchHook] Color matching failed: {e}")
            out.append(target_cpu[i])

    result = torch.stack(out, dim=0).to(target.device).to(torch.float32)
    return result.clamp(0.0, 1.0)


class ColorMatchDetailerHook:
    """
    Detailer hook that applies color matching to enhanced images.
    Matches the enhanced/inpainted crop to the original crop's colors.
    """

    def __init__(self, method="adain", strength=1.0):
        """
        Args:
            method: Color matching method
            strength: Blend strength (1.0 = full color match, 0.0 = no change)
        """
        self.method = method
        self.strength = strength
        self.original_crop = None

    def set_steps(self, info):
        """Called by detailer with (cur_step, total_step)."""
        pass

    def post_decode(self, pixels):
        """
        Apply color matching after VAE decode.

        Args:
            pixels: Enhanced/inpainted image (B, H, W, C)

        Returns:
            Color-matched image
        """
        if self.original_crop is None:
            return pixels

        # Resize reference to match target size if needed
        reference = self.original_crop
        if reference.shape[1:3] != pixels.shape[1:3]:
            reference = torch.nn.functional.interpolate(
                reference.permute(0, 3, 1, 2),
                size=pixels.shape[1:3],
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1)

        # Apply color matching based on method
        if self.method == "adain":
            matched = adain_color_match(pixels, reference)
        else:
            # color-matcher methods
            matched = color_matcher_transfer(pixels, reference, self.method)

        # Blend with strength
        if self.strength < 1.0:
            result = matched * self.strength + pixels * (1.0 - self.strength)
        else:
            result = matched

        # Clean up
        self.original_crop = None

        return result

    def post_upscale(self, pixels, mask=None):
        """
        Save the original upscaled crop for later reference.

        Args:
            pixels: Original crop (upscaled) (B, H, W, C)
            mask: Optional mask

        Returns:
            Unchanged pixels
        """
        self.original_crop = pixels.clone()
        return pixels

    def post_encode(self, samples):
        """No-op hook."""
        return samples

    def pre_decode(self, samples):
        """No-op hook."""
        return samples

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        """No-op hook."""
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise

    def post_crop_region(self, w, h, item_bbox, crop_region):
        """No-op hook."""
        return crop_region

    def touch_scaled_size(self, w, h):
        """No-op hook."""
        return w, h

    def cycle_latent(self, latent):
        """No-op hook for DetailerHook compatibility."""
        return latent

    def post_detection(self, segs):
        """No-op hook for DetailerHook compatibility."""
        return segs

    def post_paste(self, image):
        """No-op hook for DetailerHook compatibility."""
        return image

    def get_custom_noise(self, seed, noise, is_touched):
        """No-op hook for DetailerHook compatibility."""
        return noise, is_touched

    def get_custom_sampler(self):
        """No-op hook for DetailerHook compatibility."""
        return None

    def get_skip_sampling(self):
        """No-op hook for DetailerHook compatibility."""
        return False

    def should_retry_patch(self, patch):
        """No-op hook for DetailerHook compatibility."""
        return False

    def should_skip_by_cnet_image(self, cnet_image):
        """No-op hook for DetailerHook compatibility."""
        return False


class XJSegsColorMatchHookProvider:
    """
    Create a color matching hook for SEGS detailer.

    This hook matches the colors of enhanced/inpainted crops to their original colors,
    preventing color shifts during detailing operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method": (
                    ["adain", "mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"],
                    {"default": "mkl"}
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}
                ),
            },
        }

    RETURN_TYPES = ("DETAILER_HOOK",)
    RETURN_NAMES = ("hook",)
    CATEGORY = "XJNodes/segs/hooks"
    FUNCTION = "create_hook"

    def create_hook(self, method, strength):
        """
        Create color matching detailer hook.

        Args:
            method: Color matching method
                - adain: Fast, matches mean/std statistics (no dependencies)
                - mkl: Monge-Kantorovitch Linear (requires color-matcher)
                - hm: Histogram Matching (requires color-matcher)
                - reinhard: Reinhard method (requires color-matcher)
                - mvgd: Mean Variance and Gradient Deviation (requires color-matcher)
                - hm-mvgd-hm: Combined method (requires color-matcher)
                - hm-mkl-hm: Combined method (requires color-matcher)
            strength: Blend strength (1.0 = full match, 0.0 = no change)

        Returns:
            Detailer hook instance
        """
        hook = ColorMatchDetailerHook(method=method, strength=strength)
        return (hook,)


NODE_CLASS_MAPPINGS = {
    "XJSegsColorMatchHookProvider": XJSegsColorMatchHookProvider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSegsColorMatchHookProvider": "SEGS Color Match Hook",
}
