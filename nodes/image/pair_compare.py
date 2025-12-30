"""
Image Pair Compare Node - Compares two image sets with paired selection.

Features:
- Numeric tabs (1, 2, 3...) to select corresponding A/B image pairs
- Slide and Click comparison modes (like rgthree)
- Image size display under each image
- Validates that both inputs have the same number of images
"""

import nodes


class XJImagePairCompare(nodes.PreviewImage):
    """A node that compares paired images with numeric tabs."""

    NAME = "Image Pair Compare"
    CATEGORY = "XJNodes/image"
    FUNCTION = "compare_pairs"
    DESCRIPTION = "Compares paired images with numeric selection tabs"

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
        return {
            "required": {},
            "optional": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def compare_pairs(
        self,
        images_a=None,
        images_b=None,
        filename_prefix="xj.compare.",
        prompt=None,
        extra_pnginfo=None,
    ):
        # Validate: both inputs must have same count
        count_a = len(images_a) if images_a is not None else 0
        count_b = len(images_b) if images_b is not None else 0

        if count_a > 0 and count_b > 0 and count_a != count_b:
            raise ValueError(
                f"Image count mismatch: images_a has {count_a} image(s), "
                f"images_b has {count_b} image(s). Both must have the same number of images."
            )

        if count_a == 0 or count_b == 0:
            raise ValueError(
                f"Both inputs must be provided. Got: images_a={count_a > 0}, images_b={count_b > 0}"
            )

        result = {"ui": {"a_images": [], "b_images": []}}
        if images_a is not None and len(images_a) > 0:
            result["ui"]["a_images"] = self.save_images(
                images_a, filename_prefix, prompt, extra_pnginfo
            )["ui"]["images"]

        if images_b is not None and len(images_b) > 0:
            result["ui"]["b_images"] = self.save_images(
                images_b, filename_prefix, prompt, extra_pnginfo
            )["ui"]["images"]

        return result


NODE_CLASS_MAPPINGS = {
    "XJImagePairCompare": XJImagePairCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJImagePairCompare": "Image Pair Compare",
}
