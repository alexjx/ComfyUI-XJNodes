"""
SEGS filter nodes - select subsets of SEGS.
"""

from .core import create_segs, get_segs_shape, get_segs_list


class XJSegsFilter:
    """Filter SEGS by various criteria."""

    FILTER_MODES = [
        "by_area",
        "by_width",
        "by_height",
        "by_confidence",
        "top_n",
        "bottom_n",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "mode": (cls.FILTER_MODES,),
                "threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1},
                ),
            },
            "optional": {
                "order": (["ascending", "descending"], {"default": "descending"}),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("filtered_segs",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "filter"

    def filter(self, segs, mode, threshold, order="descending"):
        shape = get_segs_shape(segs)
        seg_list = get_segs_list(segs)

        if not seg_list:
            return (segs,)

        filtered = []

        if mode == "by_area":
            for seg in seg_list:
                x1, y1, x2, y2 = seg.crop_region
                area = (x2 - x1) * (y2 - y1)
                if area >= threshold:
                    filtered.append(seg)

        elif mode == "by_width":
            for seg in seg_list:
                x1, y1, x2, y2 = seg.crop_region
                width = x2 - x1
                if width >= threshold:
                    filtered.append(seg)

        elif mode == "by_height":
            for seg in seg_list:
                x1, y1, x2, y2 = seg.crop_region
                height = y2 - y1
                if height >= threshold:
                    filtered.append(seg)

        elif mode == "by_confidence":
            for seg in seg_list:
                if seg.confidence >= threshold:
                    filtered.append(seg)

        elif mode == "top_n":
            n = int(threshold)
            # Sort by some criteria first (by confidence or position)
            if order == "descending":
                sorted_segs = sorted(seg_list, key=lambda s: s.confidence, reverse=True)
            else:
                sorted_segs = sorted(seg_list, key=lambda s: s.confidence)
            filtered = sorted_segs[:n]

        elif mode == "bottom_n":
            n = int(threshold)
            if order == "descending":
                sorted_segs = sorted(seg_list, key=lambda s: s.confidence, reverse=True)
            else:
                sorted_segs = sorted(seg_list, key=lambda s: s.confidence)
            filtered = sorted_segs[-n:]

        return (create_segs(shape, filtered),)


class XJSegsFilterByLabel:
    """Filter SEGS by label(s)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "labels": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "mode": (["include", "exclude"], {"default": "include"}),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("filtered_segs",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "filter_by_label"

    def filter_by_label(self, segs, labels, mode="include"):
        shape = get_segs_shape(segs)
        seg_list = get_segs_list(segs)

        if not seg_list:
            return (segs,)

        # Parse labels (comma-separated)
        label_set = set(label.strip() for label in labels.split(",") if label.strip())

        if not label_set:
            # No labels specified, return all or none based on mode
            if mode == "include":
                return (create_segs(shape, []),)
            else:
                return (segs,)

        filtered = []
        for seg in seg_list:
            seg_label = seg.label if seg.label else ""
            if mode == "include":
                if seg_label in label_set:
                    filtered.append(seg)
            else:  # exclude
                if seg_label not in label_set:
                    filtered.append(seg)

        return (create_segs(shape, filtered),)


NODE_CLASS_MAPPINGS = {
    "XJSegsFilter": XJSegsFilter,
    "XJSegsFilterByLabel": XJSegsFilterByLabel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSegsFilter": "SEGS Filter",
    "XJSegsFilterByLabel": "SEGS Filter By Label",
}
