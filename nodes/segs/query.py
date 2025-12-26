"""
SEGS query nodes - inspect SEGS without modification.
"""

from .core import count_segs, get_segs_list


class XJSegsCount:
    """Count the number of segments in SEGS."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "count"

    def count(self, segs):
        count = count_segs(segs)
        return (count,)


class XJSegsPick:
    """Pick a single SEG from SEGS by index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("SEG",)
    RETURN_NAMES = ("seg",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "pick"

    def pick(self, segs, index):
        seg_list = get_segs_list(segs)

        if not seg_list:
            raise ValueError("SEGS is empty, cannot pick segment")

        if index < 0 or index >= len(seg_list):
            raise ValueError(
                f"Index {index} out of range. SEGS contains {len(seg_list)} segments (0-{len(seg_list) - 1})"
            )

        return (seg_list[index],)


NODE_CLASS_MAPPINGS = {
    "XJSegsCount": XJSegsCount,
    "XJSegsPick": XJSegsPick,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSegsCount": "SEGS Count",
    "XJSegsPick": "SEGS Pick",
}
