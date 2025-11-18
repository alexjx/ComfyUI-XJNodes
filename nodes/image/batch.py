import random
import numpy as np
import torch
from PIL import Image


def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(
        np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class XJImageGrid:
    """
    Creates a image grid from batch
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_grid"
    CATEGORY = "XJNode/Image"

    def image_grid(self, images):
        batch_size = images.shape[0]
        # determine number of rows and columns by square root of batch size
        # we want to be as squre as possible
        rows = int(np.ceil(np.sqrt(batch_size)))
        columns = int(np.ceil(batch_size / rows))
        height = images.shape[1]
        width = images.shape[2]
        # create a white canvas
        canvas = Image.new(
            "RGB", (width * columns, height * rows), color=(255, 255, 255)
        )
        for i, img in enumerate(images):
            # paste image on the canvas
            row = i // columns
            column = i % columns
            canvas.paste(tensor2pil(img), (column * width, row * height))
        return pil2tensor(canvas).unsqueeze(0)


class OneImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "XJNode/util"

    def execute(self, image, index):
        if index < 1:
            index = 1
        if index > image.shape[0]:
            index = image.shape[0]
        return (image[index - 1 : index],)


class XJImageListFilter:
    RETURN_TYPES = (
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "images",
        "removed_indices",
    )
    FUNCTION = "filter"
    CATEGORY = "XJNode/image"
    DESCRIPTION = "Removes empty images from a list"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "empty_color": ("STRING", {"default": "0, 0, 0"}),
                "empty_threshold": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {},
        }

    def filter(self, images, empty_color, empty_threshold):
        input_images = [img.clone() for img in images]

        empty_color_list = [int(color.strip()) for color in empty_color[0].split(",")]
        empty_color_tensor = torch.tensor(empty_color_list, dtype=torch.float32).to(
            input_images[0].device
        )
        _empty_threshold = empty_threshold[0]

        output_images = []
        empty_indices = []
        for idx, img in enumerate(input_images):
            color_diff = torch.abs(img - empty_color_tensor)
            mean_diff = color_diff.mean(dim=(1, 2, 3))
            if mean_diff <= _empty_threshold:
                empty_indices.append(idx)
                continue
            output_images.append(img)

        return (
            output_images,
            ", ".join(str(idx) for idx in empty_indices),
        )


class XJRandomImagesFromBatch:
    """
    Select N images from a batch. You can provide a comma-separated list
    of 1-based indices in `mandatory_list` which will always be included.
    The remaining images are chosen randomly using `seed`.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "count": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "optional": {
                "mandatory_list": ("STRING", {"default": ""}),
                "exclude_list": ("STRING", {"default": ""}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select"
    CATEGORY = "XJNode/image"
    DESCRIPTION = "Select N images from a batch; `mandatory_list` is comma-separated 1-based indices."

    def select(self, images, count, mandatory_list="", exclude_list="", seed=0):
        # images: tensor with shape (batch, H, W, C)
        batch_size = int(images.shape[0])

        # Clamp count to batch size
        exclude_list = [p.strip() for p in exclude_list.split(",")]
        exclude_list_int = []
        for p in exclude_list:
            try:
                idx1 = int(p)
            except Exception:
                # skip invalid entries
                continue
            # clamp 1..batch_size and convert to 0-based
            if idx1 < 1:
                idx1 = 1
            if idx1 > batch_size:
                idx1 = batch_size
            exclude_list_int.append(idx1 - 1)
        exclude_list = set(exclude_list_int)

        # Parse mandatory list (1-based indices). Ignore non-integers.
        mand = []
        mandatory_list = mandatory_list.split(",")
        parts = [p.strip() for p in mandatory_list if p.strip() != ""]
        for p in parts:
            try:
                idx1 = int(p)
            except Exception:
                # skip invalid entries
                continue
            # clamp 1..batch_size and convert to 0-based
            if idx1 < 1:
                idx1 = 1
            if idx1 > batch_size:
                idx1 = batch_size
            mand.append(idx1 - 1)

        # Remove duplicates while preserving order
        seen = set()
        mand_unique = []
        for i in mand:
            if i not in seen:
                seen.add(i)
                mand_unique.append(i)
        mand = mand_unique

        # If mand already fills or exceeds requested count, truncate
        if len(mand) >= count:
            chosen = mand[:count]
        else:
            # Build remaining pool
            remaining = [
                i for i in range(batch_size) if i not in seen and i not in exclude_list
            ]
            rnd = random.Random(seed)
            rnd.shuffle(remaining)
            need = count - len(mand)
            if need > len(remaining):
                need = len(remaining)
            chosen = mand + remaining[:need]

        if len(chosen) == 0:
            # return empty tensor with zero batch dimension but same H/W/C
            h, w, c = images.shape[1], images.shape[2], images.shape[3]
            empty = torch.zeros((0, h, w, c), dtype=images.dtype, device=images.device)
            return (empty,)

        idx_tensor = torch.tensor(chosen, dtype=torch.long, device=images.device)
        out = images.index_select(0, idx_tensor)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "XJImageGrid": XJImageGrid,
    "XJOneImageFromBatch": OneImageFromBatch,
    "XJImageListFilter": XJImageListFilter,
    "XJRandomImagesFromBatch": XJRandomImagesFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJImageGrid": "Image Grid",
    "XJOneImageFromBatch": "One Image From Batch",
    "XJImageListFilter": "Image List Filter",
    "XJRandomImagesFromBatch": "Random Images From Batch",
}
