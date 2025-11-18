import math
import comfy.utils


class XJImageScaleCalc:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
            },
            "optional": {
                "divide_by": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "width",
        "height",
    )
    FUNCTION = "calculate"
    CATEGORY = "XJNode/Image"

    def calculate(self, image, scale, divide_by=0):
        # Get current dimensions from the input image
        _, h, w, _ = image.shape

        # Calculate new dimensions
        new_width = int(w * scale)
        new_height = int(h * scale)

        # If divide_by is specified and greater than 0, make dimensions divisible by it
        if divide_by > 0:
            new_width = ((new_width + divide_by - 1) // divide_by) * divide_by
            new_height = ((new_height + divide_by - 1) // divide_by) * divide_by

        return (
            new_width,
            new_height,
        )


class XJImageScaleMegapixel:
    """
    Scale image to target megapixels with different methods
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01},
                ),
                "round_to": (
                    [8, 16, 32, 64, 128],
                    {"default": 8},
                ),
                "method": (
                    ["fill", "fit", "crop"],
                    {"default": "fit"},
                ),
                "upscale_mode": (
                    ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"],
                    {"default": "lanczos"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_to_megapixels"
    CATEGORY = "XJNode/Image"
    DESCRIPTION = """
Scale image to target megapixels.
- fill: Scale to exactly reach target (may change aspect ratio)
- fit: Scale to fit within target (maintains aspect ratio)
- crop: Scale to cover target and crop (maintains aspect ratio)
"""

    def _round_to_multiple(self, value, multiple):
        """Round value to nearest multiple"""
        return int(round(value / multiple) * multiple)

    def scale_to_megapixels(self, image, megapixels, round_to, method, upscale_mode):
        # Get current dimensions
        batch_size, height, width, channels = image.shape
        current_pixels = height * width
        target_pixels = megapixels * 1_000_000

        # Calculate new dimensions based on method
        if method == "fill":
            # Fill: create square-ish dimensions at target megapixels
            side_length = math.sqrt(target_pixels)
            new_width = self._round_to_multiple(side_length, round_to)
            new_height = self._round_to_multiple(side_length, round_to)
            crop_mode = "disabled"

        elif method == "fit":
            # Fit: maintain aspect ratio, fit within target
            aspect_ratio = width / height
            scale = math.sqrt(target_pixels / current_pixels)
            new_width = self._round_to_multiple(width * scale, round_to)
            new_height = self._round_to_multiple(height * scale, round_to)
            crop_mode = "disabled"

        elif method == "crop":
            # Crop: maintain aspect ratio, scale to cover target, then crop
            aspect_ratio = width / height
            # Scale up to cover the target megapixels
            scale = math.sqrt(target_pixels / current_pixels)
            temp_width = width * scale
            temp_height = height * scale

            # Round to get final dimensions
            new_width = self._round_to_multiple(temp_width, round_to)
            new_height = self._round_to_multiple(temp_height, round_to)

            # Adjust one dimension to better match target megapixels
            actual_pixels = new_width * new_height
            if actual_pixels < target_pixels:
                # Need to increase size
                if aspect_ratio > 1:  # wider than tall
                    new_width = self._round_to_multiple(
                        math.sqrt(target_pixels * aspect_ratio), round_to
                    )
                    new_height = self._round_to_multiple(
                        new_width / aspect_ratio, round_to
                    )
                else:  # taller than wide
                    new_height = self._round_to_multiple(
                        math.sqrt(target_pixels / aspect_ratio), round_to
                    )
                    new_width = self._round_to_multiple(
                        new_height * aspect_ratio, round_to
                    )

            crop_mode = "center"

        # Ensure minimum dimensions
        new_width = max(round_to, new_width)
        new_height = max(round_to, new_height)

        # Convert image tensor from (B,H,W,C) to (B,C,H,W) for processing
        samples = image.movedim(-1, 1)

        # Use comfy's common_upscale function
        scaled = comfy.utils.common_upscale(
            samples, new_width, new_height, upscale_mode, crop_mode
        )

        # Convert back to (B,H,W,C)
        scaled = scaled.movedim(1, -1)

        return (scaled, new_width, new_height)


NODE_CLASS_MAPPINGS = {
    "XJImageScaleCalc": XJImageScaleCalc,
    "XJImageScaleMegapixel": XJImageScaleMegapixel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJImageScaleCalc": "Image Scale Calc",
    "XJImageScaleMegapixel": "Image Scale Megapixel",
}
