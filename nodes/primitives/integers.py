class XJIntOffset:
    """
    Adds an integer offset to the input
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("INT", {"forceInput": True}),
                "offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "add_offset"
    CATEGORY = "XJNodes/util"

    def add_offset(self, input, offset):
        return (input + offset,)


class XJIntegerIncrement:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("INT", {"forceInput": True}),
                "increment": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "increment"
    CATEGORY = "XJNodes/util"

    def increment(self, input, increment):
        return (input + increment,)


class XJIntegerDecrement:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("INT", {"forceInput": True}),
                "decrement": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "decrement"
    CATEGORY = "XJNodes/util"

    def decrement(self, input, decrement):
        return (input - decrement,)


NODE_CLASS_MAPPINGS = {
    "XJIntOffset": XJIntOffset,
    "XJIntegerIncrement": XJIntegerIncrement,
    "XJIntegerDecrement": XJIntegerDecrement,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJIntOffset": "Int Offset",
    "XJIntegerIncrement": "Integer Increment",
    "XJIntegerDecrement": "Integer Decrement",
}
