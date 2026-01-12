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


def channel_offset(image: Tensor, channel_idx: int, offset: int):
    """
    Add offset to a specific channel in an image tensor.

    Args:
        image: Image tensor (B, H, W, C) in [0, 1]
        channel_idx: Channel index (0=R, 1=G, 2=B)
        offset: Offset value (-255 to 255)

    Returns:
        Modified image tensor
    """
    if offset == 0:
        return image

    result = image.clone()
    # Convert to 0-255 range, add offset, clamp, convert back
    channel_data = result[..., channel_idx] * 255.0
    channel_data = (channel_data + offset).clamp(0.0, 255.0)
    result[..., channel_idx] = channel_data / 255.0
    return result


def rgb_to_hsv_tensor(rgb: Tensor):
    """Convert RGB tensor to HSV."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = torch.max(rgb, dim=-1)[0]
    minc = torch.min(rgb, dim=-1)[0]
    delta = maxc - minc

    # Hue
    h = torch.zeros_like(maxc)
    mask = delta != 0

    r_mask = mask & (maxc == r)
    g_mask = mask & (maxc == g)
    b_mask = mask & (maxc == b)

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4

    h = h / 6.0  # Normalize to [0, 1]

    # Saturation
    s = torch.zeros_like(maxc)
    s[maxc != 0] = delta[maxc != 0] / maxc[maxc != 0]

    # Value
    v = maxc

    return torch.stack([h, s, v], dim=-1)


def hsv_to_rgb_tensor(hsv: Tensor):
    """Convert HSV tensor to RGB."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    h = h * 6.0  # Denormalize from [0, 1] to [0, 6]

    i = torch.floor(h).long()
    f = h - i

    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    i = i % 6

    # Initialize output
    rgb = torch.zeros_like(hsv)

    # Select RGB values based on hue sector
    mask0 = (i == 0)
    mask1 = (i == 1)
    mask2 = (i == 2)
    mask3 = (i == 3)
    mask4 = (i == 4)
    mask5 = (i == 5)

    rgb[mask0] = torch.stack([v[mask0], t[mask0], p[mask0]], dim=-1)
    rgb[mask1] = torch.stack([q[mask1], v[mask1], p[mask1]], dim=-1)
    rgb[mask2] = torch.stack([p[mask2], v[mask2], t[mask2]], dim=-1)
    rgb[mask3] = torch.stack([p[mask3], q[mask3], v[mask3]], dim=-1)
    rgb[mask4] = torch.stack([t[mask4], p[mask4], v[mask4]], dim=-1)
    rgb[mask5] = torch.stack([v[mask5], p[mask5], q[mask5]], dim=-1)

    return rgb


def feather_mask(mask: Tensor, feather_pixels: int):
    """
    Apply Gaussian blur to mask edges for feathering.

    Args:
        mask: (B, H, W) tensor
        feather_pixels: radius of feathering

    Returns:
        Feathered mask (B, H, W)
    """
    if feather_pixels <= 0:
        return mask

    # Ensure mask has correct shape
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    elif len(mask.shape) == 3:
        if mask.shape[2] == 1:
            mask = mask.squeeze(-1).unsqueeze(0)
    elif len(mask.shape) == 4:
        mask = mask.squeeze(-1)

    # Convert to (B, 1, H, W) for conv2d
    mask = mask.unsqueeze(1)

    # Create Gaussian kernel
    kernel_size = feather_pixels * 2 + 1
    sigma = feather_pixels / 3.0

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
    import torch.nn.functional as F
    padding = feather_pixels
    blurred = F.conv2d(mask, kernel, padding=padding)

    # Remove channel dimension
    blurred = blurred.squeeze(1)

    # Clamp to [0, 1]
    blurred = torch.clamp(blurred, 0.0, 1.0)

    return blurred


class ColorCorrectRGBDetailerHook:
    """
    Detailer hook that applies RGB color correction to enhanced images.
    Adjusts red, green, blue channels independently after inpainting.
    """

    def __init__(self, red=0, green=0, blue=0, feather=0):
        self.red = red
        self.green = green
        self.blue = blue
        self.feather = feather
        self.saved_mask = None

    def set_steps(self, info):
        pass

    def post_decode(self, pixels):
        """
        Apply RGB color correction after VAE decode.

        Args:
            pixels: Enhanced/inpainted image (B, H, W, C)

        Returns:
            Color-corrected image
        """
        # Apply color correction
        corrected = pixels

        if self.red != 0:
            corrected = channel_offset(corrected, 0, self.red)
        if self.green != 0:
            corrected = channel_offset(corrected, 1, self.green)
        if self.blue != 0:
            corrected = channel_offset(corrected, 2, self.blue)

        # Blend with mask if available
        if self.saved_mask is not None:
            # Resize mask to match pixels if needed
            import torch.nn.functional as F
            if self.saved_mask.shape[1:3] != pixels.shape[1:3]:
                mask_resized = self.saved_mask.unsqueeze(1)  # (B, 1, H, W)
                mask_resized = F.interpolate(
                    mask_resized,
                    size=pixels.shape[1:3],
                    mode='bilinear',
                    align_corners=False
                )
                mask_resized = mask_resized.squeeze(1)  # (B, H, W)
            else:
                mask_resized = self.saved_mask

            # Apply feathering
            blend_mask = feather_mask(mask_resized, self.feather)

            # Expand mask to match image channels (B, H, W) -> (B, H, W, 1)
            blend_mask = blend_mask.unsqueeze(-1)

            # Blend: corrected where mask=1, original where mask=0
            result = corrected * blend_mask + pixels * (1 - blend_mask)

            # Clean up
            self.saved_mask = None
        else:
            result = corrected

        return result

    def post_upscale(self, pixels, mask=None):
        """Save mask for later use in post_decode."""
        if mask is not None:
            self.saved_mask = mask.clone()
        return pixels

    def post_encode(self, samples):
        return samples

    def pre_decode(self, samples):
        return samples

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise

    def post_crop_region(self, w, h, item_bbox, crop_region):
        return crop_region

    def touch_scaled_size(self, w, h):
        return w, h

    def cycle_latent(self, latent):
        return latent

    def post_detection(self, segs):
        return segs

    def post_paste(self, image):
        return image

    def get_custom_noise(self, seed, noise, is_touched):
        return noise, is_touched

    def get_custom_sampler(self):
        return None

    def get_skip_sampling(self):
        return False

    def should_retry_patch(self, patch):
        return False

    def should_skip_by_cnet_image(self, cnet_image):
        return False


class ColorCorrectHSVDetailerHook:
    """
    Detailer hook that applies HSV color correction to enhanced images.
    Adjusts hue, saturation, value independently after inpainting.
    """

    def __init__(self, hue=0, saturation=0, value=0, feather=0):
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.feather = feather
        self.saved_mask = None

    def set_steps(self, info):
        pass

    def post_decode(self, pixels):
        """
        Apply HSV color correction after VAE decode.

        Args:
            pixels: Enhanced/inpainted image (B, H, W, C)

        Returns:
            Color-corrected image
        """
        if self.hue == 0 and self.saturation == 0 and self.value == 0:
            corrected = pixels
        else:
            # Convert to HSV
            hsv = rgb_to_hsv_tensor(pixels)

            # Apply adjustments
            if self.hue != 0:
                # Hue wraps around [0, 1]
                hsv[..., 0] = (hsv[..., 0] + self.hue / 255.0) % 1.0

            if self.saturation != 0:
                hsv[..., 1] = (hsv[..., 1] + self.saturation / 255.0).clamp(0.0, 1.0)

            if self.value != 0:
                hsv[..., 2] = (hsv[..., 2] + self.value / 255.0).clamp(0.0, 1.0)

            # Convert back to RGB
            corrected = hsv_to_rgb_tensor(hsv).clamp(0.0, 1.0)

        # Blend with mask if available
        if self.saved_mask is not None:
            # Resize mask to match pixels if needed
            import torch.nn.functional as F
            if self.saved_mask.shape[1:3] != pixels.shape[1:3]:
                mask_resized = self.saved_mask.unsqueeze(1)  # (B, 1, H, W)
                mask_resized = F.interpolate(
                    mask_resized,
                    size=pixels.shape[1:3],
                    mode='bilinear',
                    align_corners=False
                )
                mask_resized = mask_resized.squeeze(1)  # (B, H, W)
            else:
                mask_resized = self.saved_mask

            # Apply feathering
            blend_mask = feather_mask(mask_resized, self.feather)

            # Expand mask to match image channels (B, H, W) -> (B, H, W, 1)
            blend_mask = blend_mask.unsqueeze(-1)

            # Blend: corrected where mask=1, original where mask=0
            result = corrected * blend_mask + pixels * (1 - blend_mask)

            # Clean up
            self.saved_mask = None
        else:
            result = corrected

        return result

    def post_upscale(self, pixels, mask=None):
        """Save mask for later use in post_decode."""
        if mask is not None:
            self.saved_mask = mask.clone()
        return pixels

    def post_encode(self, samples):
        return samples

    def pre_decode(self, samples):
        return samples

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise

    def post_crop_region(self, w, h, item_bbox, crop_region):
        return crop_region

    def touch_scaled_size(self, w, h):
        return w, h

    def cycle_latent(self, latent):
        return latent

    def post_detection(self, segs):
        return segs

    def post_paste(self, image):
        return image

    def get_custom_noise(self, seed, noise, is_touched):
        return noise, is_touched

    def get_custom_sampler(self):
        return None

    def get_skip_sampling(self):
        return False

    def should_retry_patch(self, patch):
        return False

    def should_skip_by_cnet_image(self, cnet_image):
        return False


class XJSegsColorCorrectRGBHookProvider:
    """
    Create an RGB color correction hook for SEGS detailer.

    Applies fixed RGB channel adjustments to enhanced/inpainted images
    before stitching back. Useful for correcting color casts or tints
    that appear after inpainting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "red": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("DETAILER_HOOK",)
    RETURN_NAMES = ("hook",)
    CATEGORY = "XJNodes/segs/hooks"
    FUNCTION = "create_hook"

    def create_hook(self, red, green, blue, feather):
        """
        Create RGB color correction detailer hook.

        Args:
            red: Red channel offset (-255 to 255)
            green: Green channel offset (-255 to 255)
            blue: Blue channel offset (-255 to 255)
            feather: Feathering radius in pixels (0 = no feather)

        Returns:
            Detailer hook instance
        """
        hook = ColorCorrectRGBDetailerHook(red=red, green=green, blue=blue, feather=feather)
        return (hook,)


class XJSegsColorCorrectHSVHookProvider:
    """
    Create an HSV color correction hook for SEGS detailer.

    Applies fixed HSV adjustments to enhanced/inpainted images
    before stitching back. Useful for adjusting color tone, saturation,
    or brightness after inpainting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hue": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "saturation": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "value": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("DETAILER_HOOK",)
    RETURN_NAMES = ("hook",)
    CATEGORY = "XJNodes/segs/hooks"
    FUNCTION = "create_hook"

    def create_hook(self, hue, saturation, value, feather):
        """
        Create HSV color correction detailer hook.

        Args:
            hue: Hue shift (-255 to 255, wraps around)
            saturation: Saturation offset (-255 to 255)
            value: Brightness offset (-255 to 255)
            feather: Feathering radius in pixels (0 = no feather)

        Returns:
            Detailer hook instance
        """
        hook = ColorCorrectHSVDetailerHook(hue=hue, saturation=saturation, value=value, feather=feather)
        return (hook,)


NODE_CLASS_MAPPINGS = {
    "XJSegsColorMatchHookProvider": XJSegsColorMatchHookProvider,
    "XJSegsColorCorrectRGBHookProvider": XJSegsColorCorrectRGBHookProvider,
    "XJSegsColorCorrectHSVHookProvider": XJSegsColorCorrectHSVHookProvider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSegsColorMatchHookProvider": "SEGS Color Match Hook",
    "XJSegsColorCorrectRGBHookProvider": "SEGS Color Correct RGB Hook",
    "XJSegsColorCorrectHSVHookProvider": "SEGS Color Correct HSV Hook",
}
