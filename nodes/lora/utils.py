"""Utility classes for dynamic lora loading"""

from typing import Union


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    """A special class to make flexible nodes that accept dynamic inputs.

    Enables dynamic number of inputs (like for Power Lora Loader).
    """

    def __init__(self, type, data: Union[dict, None] = None):
        """Initializes the FlexibleOptionalInputType.

        Args:
            type: The flexible type to use when ComfyUI retrieves an unknown key.
            data: An optional dict to use as the basis. These are the starting optional node types.
        """
        self.type = type
        self.data = data
        if self.data is not None:
            for k, v in self.data.items():
                self[k] = v

    def __getitem__(self, key):
        # If we have this key in the initial data, then return it. Otherwise return the tuple with our
        # flexible type.
        if self.data is not None and key in self.data:
            val = self.data[key]
            return val
        return (self.type,)

    def __contains__(self, key):
        """Always contain a key, and we'll always return the tuple above when asked for it."""
        return True


any_type = AnyType("*")
