"""
List manipulation nodes for XJNodes

Provides utilities for building and manipulating lists/batches of data.
Image lists are implemented as batched tensors for ComfyUI compatibility.
"""

import torch


# Generic type that accepts anything
class AlwaysEqualProxy(str):
    """Type wildcard that matches any type"""

    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


any_type = AlwaysEqualProxy("*")


class XJEmptyImageList:
    """
    Creates an empty image list.

    Returns an empty Python list that can be appended to.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "create_empty"
    CATEGORY = "XJNodes/Lists"

    def create_empty(self):
        """
        Return an empty Python list.
        With OUTPUT_IS_LIST = (True,), this is treated as a list by ComfyUI.
        """
        return ([],)


class XJAppendImageList:
    """
    Appends an image to an image list.

    Takes a Python list of images and adds a new image to it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("list",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "append"
    CATEGORY = "XJNodes/Lists"

    def append(self, image_list, image):
        """
        Append image to list.

        Args:
            image_list: Python list of IMAGE tensors (from OUTPUT_IS_LIST=True source)
            image: IMAGE tensor or [tensor] (from INPUT_IS_LIST wrapping)

        Returns:
            New Python list with image(s) appended

        Note:
            If image is a batch tensor (batch_size > 1), it will be split into
            individual images and each will be appended separately.
        """
        # Validate image_list is actually a list
        if not isinstance(image_list, list):
            raise ValueError(f"image_list must be a list, got {type(image_list)}")

        # Step 1: Unwrap image from INPUT_IS_LIST wrapping if present
        # Normal nodes output tensor → wrapped as [tensor] by ComfyUI → we receive [tensor]
        unwrapped_image = image
        if isinstance(image, list):
            if len(image) == 1:
                unwrapped_image = image[0]
            elif len(image) == 0:
                # Empty list - nothing to append
                return (image_list,)
            else:
                raise ValueError(
                    f"Expected single image or [image], got list with {len(image)} elements"
                )

        # Step 2: Check if the unwrapped image is a batch tensor
        # IMAGE tensors are 4D: (batch, height, width, channels)
        if isinstance(unwrapped_image, torch.Tensor) and unwrapped_image.dim() == 4:
            batch_size = unwrapped_image.size(0)
            if batch_size > 1:
                # Multiple images in batch - split into individual images
                for i in range(batch_size):
                    # Preserve batch dimension using slicing
                    image_list.append(unwrapped_image[i : i + 1])
            else:
                # Single image with batch dimension - append as-is
                image_list.append(unwrapped_image)
        else:
            # Not a 4D tensor - append as-is
            image_list.append(unwrapped_image)

        return (image_list,)


class XJImageListLength:
    """
    Returns the number of images in an image list.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    INPUT_IS_LIST = True
    FUNCTION = "get_length"
    CATEGORY = "XJNodes/Lists"

    def get_length(self, image_list):
        """
        Get the length of the image list.

        Args:
            image_list: Python list of IMAGE tensors (from OUTPUT_IS_LIST source)

        Returns:
            Number of images in the list
        """
        # INPUT_IS_LIST receives list directly from OUTPUT_IS_LIST
        if not isinstance(image_list, list):
            raise ValueError(f"Expected list, got {type(image_list)}")

        return (len(image_list),)


class XJGetImageFromList:
    """
    Retrieves a single image from an image list by index.

    Supports negative indexing (e.g., -1 for last image).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",),
                "index": (
                    "INT",
                    {"default": 0, "min": -10000, "max": 10000, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    INPUT_IS_LIST = True
    FUNCTION = "get_image"
    CATEGORY = "XJNodes/Lists"

    def get_image(self, image_list, index):
        """
        Extract a single image from the list at the specified index.

        Args:
            image_list: Python list of IMAGE tensors (from OUTPUT_IS_LIST source)
            index: Index to retrieve (wrapped as [value] from INPUT_IS_LIST)

        Returns:
            Single IMAGE tensor from the list
        """
        # Validate image_list is actually a list
        if not isinstance(image_list, list):
            raise ValueError(f"Expected list, got {type(image_list)}")

        if len(image_list) == 0:
            raise ValueError("Cannot get image from empty list")

        # Unwrap index from INPUT_IS_LIST wrapping and cast to int
        if isinstance(index, list):
            if len(index) == 0:
                raise ValueError("Index parameter is empty")
            index = index[0]

        # Cast to int to ensure proper type
        index = int(index)

        # Python list indexing handles negative indices automatically
        # Just do bounds check
        if index < -len(image_list) or index >= len(image_list):
            raise ValueError(
                f"Index {index} out of range for list of size {len(image_list)}"
            )

        # Return the image at the index
        return (image_list[index],)


class XJImageListSlice:
    """
    Extracts a slice (range) of images from an image list.

    Similar to Python's list slicing: list[start:end]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",),
                "start": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "end": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("list",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "slice_list"
    CATEGORY = "XJNodes/Lists"

    def slice_list(self, image_list, start, end):
        """
        Extract a range of images from the list.

        Args:
            image_list: Python list of IMAGE tensors (from OUTPUT_IS_LIST source)
            start: Starting index (wrapped as [value] from INPUT_IS_LIST)
            end: Ending index (wrapped as [value] from INPUT_IS_LIST), -1 means end of list

        Returns:
            Sliced Python list
        """
        # Validate image_list is actually a list
        if not isinstance(image_list, list):
            raise ValueError(f"Expected list, got {type(image_list)}")

        if len(image_list) == 0:
            raise ValueError("Cannot slice empty list")

        # Unwrap start and end from INPUT_IS_LIST wrapping and cast to int
        if isinstance(start, list):
            if len(start) == 0:
                raise ValueError("Start parameter is empty")
            start = start[0]
        if isinstance(end, list):
            if len(end) == 0:
                raise ValueError("End parameter is empty")
            end = end[0]

        # Cast to int to ensure proper type
        start = int(start)
        end = int(end)

        # Handle end=-1 as "to the end"
        if end == -1:
            end = len(image_list)

        # Bounds check
        if start < 0 or start >= len(image_list):
            raise ValueError(
                f"Start index {start} out of range for list of size {len(image_list)}"
            )
        if end < start or end > len(image_list):
            raise ValueError(
                f"End index {end} invalid for list of size {len(image_list)}"
            )

        return (image_list[start:end],)


class XJWrapAsList:
    """
    Wraps a single value in a list.

    Useful for converting single items to lists before passing to loop nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (any_type,),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "wrap"
    CATEGORY = "XJNodes/Lists"

    def wrap(self, value):
        """
        Wrap a single value in a list.

        Args:
            value: Any single value

        Returns:
            List containing the single value
        """
        return ([value],)


class XJUnwrapFromList:
    """
    Extracts a single value from a ComfyUI-wrapped list.

    Without OUTPUT_IS_LIST, ComfyUI wraps outputs as [value].
    With INPUT_IS_LIST=True, nodes receive [value] directly.
    This node unwraps to get the value for non-INPUT_IS_LIST consumers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (any_type,),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    INPUT_IS_LIST = True
    FUNCTION = "unwrap"
    CATEGORY = "XJNodes/Lists"

    def unwrap(self, value):
        """
        Extract value from a single-element list.

        Handles potential double-wrapping from INPUT_IS_LIST + OUTPUT_IS_LIST chain:
        - Single wrap: [value] → value
        - Double wrap: [[value]] → value (unwraps twice)

        Args:
            value: List with exactly one item, possibly double-wrapped

        Returns:
            The unwrapped value
        """
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value)}")

        if len(value) != 1:
            raise ValueError(f"Expected exactly 1 item in list, got {len(value)} items")

        # First unwrap
        unwrapped = value[0]

        # Check for double-wrapping (from INPUT_IS_LIST + OUTPUT_IS_LIST chain)
        # If it's still a list with exactly one element, unwrap again
        if isinstance(unwrapped, list) and len(unwrapped) == 1:
            unwrapped = unwrapped[0]

        return (unwrapped,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "XJEmptyImageList": XJEmptyImageList,
    "XJAppendImageList": XJAppendImageList,
    "XJImageListLength": XJImageListLength,
    "XJGetImageFromList": XJGetImageFromList,
    "XJImageListSlice": XJImageListSlice,
    "XJWrapAsList": XJWrapAsList,
    "XJUnwrapFromList": XJUnwrapFromList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJEmptyImageList": "Empty Image List (XJ)",
    "XJAppendImageList": "Append to Image List (XJ)",
    "XJImageListLength": "Image List Length (XJ)",
    "XJGetImageFromList": "Get Image from List (XJ)",
    "XJImageListSlice": "Image List Slice (XJ)",
    "XJWrapAsList": "Wrap as List (XJ)",
    "XJUnwrapFromList": "Unwrap from List (XJ)",
}
