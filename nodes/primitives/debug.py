"""Debug node for logging and passthrough"""

import logging

logger = logging.getLogger(__name__)


class XJDebug:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*",),
                "text": ("STRING", {"default": "Debug", "multiline": False}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    CATEGORY = "XJNodes/util"
    FUNCTION = "debug"
    DESCRIPTION = """
Debug node that logs a message and passes the input through unchanged.
"""

    def debug(self, value, text):
        print(f"\033[96m[XJDebug] {text}\033[0m")
        return (value,)


NODE_CLASS_MAPPINGS = {
    "XJDebug": XJDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJDebug": "Debug",
}
