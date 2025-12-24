import numpy as np


class XJFloatRangeList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "end": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "step": ("FLOAT", {"default": 0.1, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("floats",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "range_list"
    CATEGORY = "XJNodes/util"

    def range_list(self, start, end, step):
        return (np.arange(start, end, step).tolist(),)


NODE_CLASS_MAPPINGS = {
    "XJFloatRangeList": XJFloatRangeList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJFloatRangeList": "Float Range List",
}
