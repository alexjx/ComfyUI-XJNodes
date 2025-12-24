"""
Context Manager Nodes - Passthrough nodes with boolean outputs for workflow control

These nodes are simple passthrough utilities that provide boolean constants.
You can use them to establish "context boundaries" in your workflow by choosing
which boolean output to connect to downstream nodes.

Usage Example (SAM3 segmentation):
1. Add ContextEnter node, connect its FALSE output to SAM3's "should_unload" input
2. Process multiple items with SAM3 (model stays in VRAM)
3. Add ContextExit node, connect its TRUE output to the last SAM3's "should_unload" input
4. Model unloads after the last operation

The nodes don't modify anything - you manually wire the booleans where needed.
"""


class XJContextEnter:
    """
    Context Enter Node - Passthrough with boolean outputs.

    Takes any input, passes it through, and provides TRUE and FALSE boolean outputs.
    For "context enter" semantic, connect the FALSE output to "should_unload" inputs
    (meaning: don't unload, keep resources in memory).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "value": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("*", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("passthrough", "FALSE", "TRUE")
    FUNCTION = "passthrough"
    CATEGORY = "XJNodes/util"
    DESCRIPTION = """
Passthrough node with boolean outputs for establishing context boundaries.

Outputs:
- passthrough: Input value unchanged (any type)
- FALSE: Boolean False (connect to "should_unload" to keep resources loaded)
- TRUE: Boolean True (if you need it)

Typical usage: Connect FALSE output to SAM3's "should_unload" input at context start.
"""

    def passthrough(self, value=None):
        """
        Passthrough with both boolean values.

        Returns:
            (value, False, True): Passthrough and both boolean options
        """
        return (value, False, True)


class XJContextExit:
    """
    Context Exit Node - Passthrough with boolean outputs.

    Takes any input, passes it through, and provides TRUE and FALSE boolean outputs.
    For "context exit" semantic, connect the TRUE output to "should_unload" inputs
    (meaning: do unload, release resources from memory).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "value": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("*", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("passthrough", "TRUE", "FALSE")
    FUNCTION = "passthrough"
    CATEGORY = "XJNodes/util"
    DESCRIPTION = """
Passthrough node with boolean outputs for establishing context boundaries.

Outputs:
- passthrough: Input value unchanged (any type)
- TRUE: Boolean True (connect to "should_unload" to release resources)
- FALSE: Boolean False (if you need it)

Typical usage: Connect TRUE output to SAM3's "should_unload" input at context end.
"""

    def passthrough(self, value=None):
        """
        Passthrough with both boolean values.

        Returns:
            (value, True, False): Passthrough and both boolean options
        """
        return (value, True, False)


NODE_CLASS_MAPPINGS = {
    "XJContextEnter": XJContextEnter,
    "XJContextExit": XJContextExit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJContextEnter": "Context Enter",
    "XJContextExit": "Context Exit",
}
