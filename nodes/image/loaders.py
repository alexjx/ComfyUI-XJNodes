import os
import json
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
import comfy.utils
import folder_paths
import node_helpers


class LoadImagesFromDirBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": (
                    "INT",
                    {"default": 0, "min": -1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
                "load_always": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "recursive": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_images"

    CATEGORY = "XJNodes/image"

    valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if "load_always" in kwargs and kwargs["load_always"]:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def _load_images(self, image_files, start_index: int, image_load_cap: int = 0):
        # start at start_index
        image_files = image_files[start_index:]

        images = []
        masks = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in image_files:
            if not os.path.exists(image_path) or os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_count += 1

        if len(images) == 1:
            return (images[0], masks[0], 1)

        elif len(images) > 1:
            # sort image by size, largest first, but we need to sort by index since we need to arrange the mask as well
            # consruct a dict of index and size
            image_sizes = {}
            for i, image in enumerate(images):
                image_sizes[i] = image.shape[1] * image.shape[2]
            # sort the dict by value
            sorted_image_sizes = sorted(
                image_sizes.items(), key=lambda x: x[1], reverse=True
            )
            # sort the images and masks by index
            images = [images[i] for i, _ in sorted_image_sizes]
            masks = [masks[i] for i, _ in sorted_image_sizes]

            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(
                        image2.movedim(-1, 1),
                        image1.shape[2],
                        image1.shape[1],
                        "bilinear",
                        "center",
                    ).movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(
                            mask2.unsqueeze(0).unsqueeze(0),
                            size=(image1.shape[1], image1.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        )
                        mask2 = mask2.squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

            return (image1, mask1, len(images))

    def _load_images_recursive(
        self, directory: str, image_load_cap: int = 0, start_index: int = 0
    ):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in self.valid_extensions):
                    files.append(os.path.join(root, filename))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No valid image files found in directory '{directory}'."
            )

        files = sorted(files)
        return self._load_images(files, start_index, image_load_cap)

    def _load_images_non_recursive(
        self, directory: str, image_load_cap: int = 0, start_index: int = 0
    ):
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        dir_files = [
            f
            for f in dir_files
            if any(f.lower().endswith(ext) for ext in self.valid_extensions)
        ]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]
        return self._load_images(dir_files, start_index, image_load_cap)

    def load_images(
        self,
        directory: str,
        image_load_cap: int = 0,
        start_index: int = 0,
        load_always=False,
        recursive: bool = False,
    ):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.'")
        if recursive:
            return self._load_images_recursive(directory, image_load_cap, start_index)
        else:
            return self._load_images_non_recursive(
                directory, image_load_cap, start_index
            )


class LoadImagesFromDirList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
                "load_always": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "recursive": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE PATH")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "load_images"

    CATEGORY = "XJNodes/image"

    valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if "load_always" in kwargs and kwargs["load_always"]:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def _load_images(self, image_files, start_index: int, image_load_cap: int = 0):
        # start at start_index
        image_files = image_files[start_index:]

        images = []
        masks = []
        file_paths = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in image_files:
            if not os.path.exists(image_path) or os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            images.append(image)
            masks.append(mask)
            file_paths.append(str(image_path))
            image_count += 1

        return (images, masks, file_paths)

    def _load_images_recursive(
        self, directory: str, image_load_cap: int = 0, start_index: int = 0
    ):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in self.valid_extensions):
                    files.append(os.path.join(root, filename))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No valid image files found in directory '{directory}'."
            )

        files = sorted(files)
        return self._load_images(files, start_index, image_load_cap)

    def _load_images_non_recursive(
        self, directory: str, image_load_cap: int = 0, start_index: int = 0
    ):
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        dir_files = [
            f
            for f in dir_files
            if any(f.lower().endswith(ext) for ext in self.valid_extensions)
        ]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]
        return self._load_images(dir_files, start_index, image_load_cap)

    def load_images(
        self,
        directory: str,
        image_load_cap: int = 0,
        start_index: int = 0,
        load_always=False,
        recursive: bool = False,
    ):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.'")

        if recursive:
            return self._load_images_recursive(directory, image_load_cap, start_index)
        else:
            return self._load_images_non_recursive(
                directory, image_load_cap, start_index
            )


class XJLoadImageWithMetadata:
    """Load image with file path and metadata - supports input/output directories"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": (["input", "output"],),
                "subdirectory": ("STRING", {"default": ""}),
                "image": ([""],),  # Combo populated dynamically by JS
            }
        }

    CATEGORY = "XJNodes/image"
    RETURN_TYPES = (
        "IMAGE",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "image",
        "file_path",
        "file_name",
        "width",
        "height",
        "format",
        "metadata",
    )
    FUNCTION = "load_image"

    def load_image(self, directory, subdirectory, image):
        # Get base directory
        if directory == "input":
            base_dir = folder_paths.get_input_directory()
        else:  # output
            base_dir = folder_paths.get_output_directory()

        # Clean subdirectory: remove leading/trailing slashes
        subdirectory = subdirectory.strip().strip("/")

        # Construct full path
        if subdirectory:
            full_dir = os.path.join(base_dir, subdirectory)
        else:
            full_dir = base_dir

        image_path = os.path.join(full_dir, image)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image_converted = i.convert("RGB")

            if len(output_images) == 0:
                w = image_converted.size[0]
                h = image_converted.size[1]

            if image_converted.size[0] != w or image_converted.size[1] != h:
                continue

            image_array = np.array(image_converted).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]

            output_images.append(image_tensor)

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

        # Get metadata
        file_name = os.path.basename(image_path)
        image_format = img.format if img.format else "UNKNOWN"
        width = w if w else 0
        height = h if h else 0

        # Read all metadata from PNG text chunks as JSON
        all_metadata = {}
        if hasattr(img, "text") and img.text:
            # Return all PNG text chunks including workflow
            all_metadata = dict(img.text)

        metadata_json = json.dumps(all_metadata) if all_metadata else "{}"

        return (
            output_image,
            image_path,
            file_name,
            width,
            height,
            image_format,
            metadata_json,
        )

    @classmethod
    def IS_CHANGED(s, directory, subdirectory, image):
        # Get base directory
        if directory == "input":
            base_dir = folder_paths.get_input_directory()
        else:
            base_dir = folder_paths.get_output_directory()

        # Clean subdirectory: remove leading/trailing slashes
        subdirectory = subdirectory.strip().strip("/")

        # Construct full path
        if subdirectory:
            full_dir = os.path.join(base_dir, subdirectory)
        else:
            full_dir = base_dir

        image_path = os.path.join(full_dir, image)

        m = hashlib.sha256()
        try:
            with open(image_path, "rb") as f:
                m.update(f.read())
            return m.digest().hex()
        except:
            return ""

    @classmethod
    def VALIDATE_INPUTS(s, directory, subdirectory, image):
        # Allow empty image (e.g., when switching directories with no images)
        if not image or image == "":
            return True

        # Get base directory
        if directory == "input":
            base_dir = folder_paths.get_input_directory()
        else:
            base_dir = folder_paths.get_output_directory()

        # Clean subdirectory: remove leading/trailing slashes
        subdirectory = subdirectory.strip().strip("/")

        # Construct full path
        if subdirectory:
            full_dir = os.path.join(base_dir, subdirectory)
        else:
            full_dir = base_dir

        image_path = os.path.join(full_dir, image)

        if not os.path.exists(image_path):
            return "Invalid image file: {}".format(image)
        return True


class XJLoadImageByPath:
    """
    Simple image loader that takes directory and filename as inputs.
    No preview, just loads and returns the image.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "XJNodes/image"

    def load_image(self, directory, filename):
        """Load image from directory + filename"""
        if not directory or not filename:
            raise ValueError("Both directory and filename must be provided")

        # Construct full path
        image_path = os.path.join(directory, filename)

        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        # Convert to RGB
        image = img.convert("RGB")
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array)[None,]

        # Extract alpha channel as mask if present
        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            # Create empty mask if no alpha channel
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return (image_tensor, mask)


NODE_CLASS_MAPPINGS = {
    "XJLoadImagesFromDirBatch": LoadImagesFromDirBatch,
    "XJLoadImagesFromDirList": LoadImagesFromDirList,
    "XJLoadImageWithMetadata": XJLoadImageWithMetadata,
    "XJLoadImageByPath": XJLoadImageByPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJLoadImagesFromDirBatch": "Load Images From Dir Batch",
    "XJLoadImagesFromDirList": "Load Images From Dir List",
    "XJLoadImageWithMetadata": "Load Image With Metadata",
    "XJLoadImageByPath": "Load Image By Path",
}
