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


class XJImageBatchResize:
    """Resize each input with preserved aspect ratio, then concatenate batches."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "resize_mode": (["pad", "crop"], {"default": "pad"}),
            },
            "optional": {f"image_{i}": ("IMAGE",) for i in range(2, 9)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "batch"
    CATEGORY = "XJNodes/image"
    DESCRIPTION = (
        "Combine up to 8 images or batches in input order. Resize with preserved "
        "aspect ratio: pad adds centered black borders; crop removes centered "
        "overflow. Inputs 2–8 may be disconnected or None."
    )

    def batch(self, image_1, width, height, resize_mode, **kwargs):
        if width < 1 or height < 1:
            raise ValueError("Target width and height must be positive.")
        if resize_mode not in ("pad", "crop"):
            raise ValueError("resize_mode must be pad or crop.")
        if image_1 is None:
            raise ValueError("image_1 is required.")

        output = []
        for index in range(1, 9):
            images = image_1 if index == 1 else kwargs.get(f"image_{index}")
            if images is None:
                continue
            if images.ndim != 4 or any(size < 1 for size in images.shape[1:]):
                raise ValueError(f"image_{index} must have shape [batch, height, width, channels].")
            if images.shape[-1] != image_1.shape[-1]:
                raise ValueError("All inputs must have the same number of channels.")
            if images.shape[0] == 0:
                continue
            images = images.to(device=image_1.device, dtype=image_1.dtype)
            source_h, source_w = images.shape[1:3]
            scale = (min if resize_mode == "pad" else max)(
                width / source_w, height / source_h
            )
            resized_w = max(1, round(source_w * scale))
            resized_h = max(1, round(source_h * scale))
            samples = images.movedim(-1, 1)
            if (resized_h, resized_w) != (source_h, source_w):
                samples = torch.nn.functional.interpolate(
                    samples, size=(resized_h, resized_w), mode="bilinear",
                    align_corners=False, antialias=True,
                )
            if resize_mode == "pad":
                left = (width - resized_w) // 2
                top = (height - resized_h) // 2
                samples = torch.nn.functional.pad(
                    samples, (left, width - resized_w - left,
                              top, height - resized_h - top),
                )
            else:
                left = (resized_w - width) // 2
                top = (resized_h - height) // 2
                samples = samples[:, :, top:top + height, left:left + width]
            output.append(samples.movedim(1, -1))

        if not output:
            return (image_1.new_empty((0, height, width, image_1.shape[-1])),)
        return (torch.cat(output, dim=0),)


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
    CATEGORY = "XJNodes/image"

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
    CATEGORY = "XJNodes/util"

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
    CATEGORY = "XJNodes/image"
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
    CATEGORY = "XJNodes/image"
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
    "XJImageBatchResize": XJImageBatchResize,
    "XJImageGrid": XJImageGrid,
    "XJOneImageFromBatch": OneImageFromBatch,
    "XJImageListFilter": XJImageListFilter,
    "XJRandomImagesFromBatch": XJRandomImagesFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJImageBatchResize": "Image Batch Resize (Pad/Crop)",
    "XJImageGrid": "Image Grid",
    "XJOneImageFromBatch": "One Image From Batch",
    "XJImageListFilter": "Image List Filter",
    "XJRandomImagesFromBatch": "Random Images From Batch",
}
