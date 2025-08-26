import os
import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageEnhance, ImageOps

import comfy.samplers
import folder_paths


class XJSchedulerAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "defaultInput": False,
                    },
                ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    CATEGORY = "XJNode/Util"
    FUNCTION = "doit"

    def doit(self, scheduler):
        return (scheduler,)


class XJSamplerAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "defaultInput": False,
                    },
                ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    CATEGORY = "XJNode/Util"
    FUNCTION = "doit"

    def doit(self, sampler):
        return (sampler,)


class XJStringPass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "string": (
                    "STRING",
                    {"forceInput": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "passthrough"
    CATEGORY = "XJNode/Util"
    DESCRIPTION = """
Passes the string through without modifying it.
"""

    def passthrough(self, string=None):
        return (string,)


class XJImageScaleCalc:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
            },
            "optional": {
                "divide_by": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "width",
        "height",
    )
    FUNCTION = "calculate"
    CATEGORY = "XJNode/Image"

    def calculate(self, image, scale, divide_by=0):
        # Get current dimensions from the input image
        _, h, w, _ = image.shape

        # Calculate new dimensions
        new_width = int(w * scale)
        new_height = int(h * scale)

        # If divide_by is specified and greater than 0, make dimensions divisible by it
        if divide_by > 0:
            new_width = ((new_width + divide_by - 1) // divide_by) * divide_by
            new_height = ((new_height + divide_by - 1) // divide_by) * divide_by

        return (
            new_width,
            new_height,
        )


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


class XJIntOffset:
    """
    Adds an integer offset to the input
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("INT", {"forceInput": True}),
                "offset": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "add_offset"
    CATEGORY = "XJNode/Util"

    def add_offset(self, input, offset):
        return (input + offset,)


class XJImageRandomTransform:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "repeat": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "distortion": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "rotation": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 180.0,
                        "step": 1.0,
                    },
                ),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "hue": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "horizon_flip": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(
        self,
        image,
        seed,
        repeat,
        distortion,
        rotation,
        brightness,
        contrast,
        saturation,
        hue,
        scale,
        horizon_flip,
    ):
        h, w = image.shape[1:3]
        image = image.repeat(repeat, 1, 1, 1).permute([0, 3, 1, 2])

        torch.manual_seed(seed)

        out = []
        for idx, img in enumerate(image):
            if idx > 0:
                transforms = []
                if distortion > 0:
                    transforms.append(
                        T.RandomPerspective(distortion_scale=distortion, p=1.0)
                    )
                if rotation > 0:
                    transforms.append(
                        T.RandomRotation(
                            degrees=rotation,
                            interpolation=T.InterpolationMode.BILINEAR,
                            expand=True,
                        )
                    )
                if brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
                    transforms.append(
                        T.ColorJitter(
                            brightness=brightness,
                            contrast=contrast,
                            saturation=saturation,
                            hue=(-hue, hue),
                        )
                    )
                if horizon_flip > 0.0:
                    transforms.append(T.RandomHorizontalFlip(p=horizon_flip))
                transforms.append(
                    T.RandomResizedCrop(
                        (h, w),
                        scale=(1 - scale, 1 + scale),
                        ratio=(w / h, w / h),
                        interpolation=T.InterpolationMode.BICUBIC,
                    )
                )
                transforms = T.Compose(transforms)
                out.append(transforms(img.unsqueeze(0)))
            else:
                out.append(img.unsqueeze(0))  # keep the first image as is

        out = torch.cat(out, dim=0).permute([0, 2, 3, 1]).clamp(0, 1)

        return (out,)


class XJImageTransform:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "repeat": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "distortion": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "rotation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 180.0,
                        "step": 1.0,
                    },
                ),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def _find_coeffs(self, pa, pb):
        # helper to compute perspective transform coefficients for PIL
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.array(matrix, dtype=np.float64)
        B = np.array(pb).reshape(8)
        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return res.tolist()

    def execute(
        self,
        image,
        seed,
        repeat,
        distortion,
        rotation,
        brightness,
        contrast,
        saturation,
    ):
        # image: tensor (B, H, W, C)
        H, W = image.shape[1:3]

        # repeat images to create candidates
        orig_B = image.shape[0]
        imgs = image.repeat(repeat, 1, 1, 1)  # (B*repeat, H, W, C)

        out_list = []

        src_quad = [(0, 0), (W, 0), (W, H), (0, H)]

        for idx in range(imgs.shape[0]):
            timg = imgs[idx]
            # convert to PIL
            arr = (timg.cpu().numpy() * 255.0).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            pil = Image.fromarray(arr)

            # per-candidate RNG
            rnd_c = random.Random(int(seed) + int(idx))

            # apply per-candidate rotation if requested
            if rotation and rotation != 0:
                # sample deterministic angle in [-rotation, rotation]
                angle = rnd_c.uniform(-float(rotation), float(rotation))
                pil = pil.rotate(angle, resample=Image.BICUBIC, expand=True)

            # apply per-candidate perspective distortion if requested
            if distortion and distortion > 0:
                max_off = float(distortion) * min(W, H) * 0.25
                dst_quad = []
                for x, y in src_quad:
                    dx = rnd_c.uniform(-1.0, 1.0) * max_off
                    dy = rnd_c.uniform(-1.0, 1.0) * max_off
                    dst_quad.append((x + dx, y + dy))
                coeffs = self._find_coeffs(src_quad, dst_quad)
                pil = pil.transform(
                    pil.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC
                )

            # flip whole repeat-groups (e.g. 1a..Na, 1b..Nb). compute repeat_group using original batch size
            repeat_group = idx // orig_B
            if (repeat_group % 2) == 1:
                pil = ImageOps.mirror(pil)

            # ensure canvas back to original size
            pil = pil.resize((W, H), resample=Image.BICUBIC)

            # Apply color adjustments deterministically if non-zero
            if brightness != 0:
                enh = ImageEnhance.Brightness(pil)
                pil = enh.enhance(1.0 + float(brightness))
            if contrast != 0:
                enh = ImageEnhance.Contrast(pil)
                pil = enh.enhance(1.0 + float(contrast))
            if saturation != 0:
                enh = ImageEnhance.Color(pil)
                pil = enh.enhance(1.0 + float(saturation))

            # convert back to tensor in range 0..1, shape (1, H, W, C)
            t = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)
            out_list.append(t)

        out = torch.cat(out_list, dim=0)
        return (out,)


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
    CATEGORY = "XJNode/Util"

    def range_list(self, start, end, step):
        return (np.arange(start, end, step).tolist(),)


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
    CATEGORY = "XJNode/Util"

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
    CATEGORY = "XJNode/Util"

    def decrement(self, input, decrement):
        return (input - decrement,)


class XJSupirParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": (
                    "INT",
                    {
                        "forceInput": True,
                    },
                ),
                "iterations": (
                    "INT",
                    {
                        "forceInput": True,
                    },
                ),
                "final_control_scale_start": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01},
                ),
                "final_control_scale_end": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01},
                ),
                "control_scale_step": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.00, "max": 1.00, "step": 0.01},
                ),
                "final_s_noise": (
                    "FLOAT",
                    {"default": 1.003, "min": 1.000, "max": 1.100, "step": 0.001},
                ),
                "s_noise_step": (
                    "FLOAT",
                    {"default": 0.001, "min": -0.100, "max": 0.100, "step": 0.001},
                ),
                "final_EDM_s_churn": (
                    "INT",
                    {"default": 3, "min": 0, "max": 40, "step": 1},
                ),
                "EDM_s_churn_step": (
                    "INT",
                    {"default": 1, "min": -40, "max": 40, "step": 1},
                ),
            },
        }

    RETURN_NAMES = (
        "control_scale_start",
        "control_scale_end",
        "s_noise",
        "EDM_s_churn",
    )
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "INT")
    FUNCTION = "calculate"

    def calculate(
        self,
        index,
        iterations,
        final_control_scale_start,
        final_control_scale_end,
        control_scale_step,
        final_s_noise,
        s_noise_step,
        final_EDM_s_churn,
        EDM_s_churn_step,
    ):
        offset = iterations - index - 1
        if offset < 0:
            offset = 0
        # Calculate the values based on the offset
        control_scale_start = final_control_scale_start - (control_scale_step * offset)
        if control_scale_start < 0:
            control_scale_start = 0
        control_scale_end = final_control_scale_end - (control_scale_step * offset)
        if control_scale_end < 0:
            control_scale_end = 0
        s_noise = final_s_noise - (s_noise_step * offset)
        if s_noise_step > 0:
            if s_noise < 1.0:
                s_noise = 1.0
        elif s_noise_step < 0:
            if s_noise > 1.1:
                s_noise = 1.1
        EDM_s_churn = final_EDM_s_churn - (EDM_s_churn_step * offset)
        if EDM_s_churn_step > 0:
            if EDM_s_churn < 0:
                EDM_s_churn = 0
        elif EDM_s_churn_step < 0:
            if EDM_s_churn > 40:
                EDM_s_churn = 40
        return (control_scale_start, control_scale_end, s_noise, EDM_s_churn)


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

    CATEGORY = "XJNode/image"

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
            if os.path.isdir(image_path) and os.path.ex:
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

    CATEGORY = "XJNode/image"

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
            if os.path.isdir(image_path) and os.path.ex:
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


class XJRandomTextFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multiline_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "text",
                    },
                ),
                "type": (("fixed", "random"),),
                "choice": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "step": 1,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "make_list"
    CATEGORY = "XJNode/text"

    def make_list(self, multiline_text, type, choice, seed):
        # Split the multiline text into a list of strings
        text_list = multiline_text.splitlines()
        # Remove empty lines
        text_list = [text.strip().lstrip("- ") for text in text_list]
        text_list = [
            text for text in text_list if text and not text.startswith("#")
        ]  # skip comments
        if not text_list:
            return ([""],)

        if type == "fixed":
            # If type is fixed, return the first 'choice' number of strings
            if choice >= len(text_list):
                raise Exception(f"Choice {choice} exceeded max length {len(text_list)}")
            selected_text = text_list[choice - 1]
        else:
            # Set the random seed for reproducibility
            random.seed(seed)
            # Randomly select one string from the list
            selected_text = random.choice(text_list)

        return (selected_text,)


class XJRandomTextFromFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "default": "outfits.md",
                    },
                ),
                "type": (("fixed", "random"),),
                "choice": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "step": 1,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "make_list"
    CATEGORY = "XJNode/text"

    def make_list(self, file_path, type, choice, seed):
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            # Read the file and split it into lines
            text_list = f.read().splitlines()
        # Remove empty lines
        text_list = [text.strip().lstrip("- ") for text in text_list]
        text_list = [
            text for text in text_list if text and not text.startswith("#")
        ]  # skip comments
        if not text_list:
            return ([""],)

        if type == "fixed":
            # If type is fixed, return the first 'choice' number of strings
            if choice >= len(text_list):
                raise Exception(f"Choice {choice} exceeded max length {len(text_list)}")
            selected_text = text_list[choice - 1]
        else:
            # Set the random seed for reproducibility
            random.seed(seed)
            # Randomly select one string from the list
            selected_text = random.choice(text_list)

        return (selected_text,)


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

    def select(self, images, count, mandatory_list="", seed=0):
        # images: tensor with shape (batch, H, W, C)
        batch_size = int(images.shape[0])

        # Parse mandatory list (1-based indices). Ignore non-integers.
        mand = []
        if mandatory_list:
            parts = [p.strip() for p in mandatory_list.split(",") if p.strip() != ""]
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
            remaining = [i for i in range(batch_size) if i not in seen]
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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "XJSchedulerAdapter": XJSchedulerAdapter,
    "XJSamplerAdapter": XJSamplerAdapter,
    "XJStringPass": XJStringPass,
    "XJImageScaleCalc": XJImageScaleCalc,
    "XJImageGrid": XJImageGrid,
    "XJIntOffset": XJIntOffset,
    "XJImageRandomTransform": XJImageRandomTransform,
    "XJImageTransform": XJImageTransform,
    "XJFloatRangeList": XJFloatRangeList,
    "XJIntegerIncrement": XJIntegerIncrement,
    "XJIntegerDecrement": XJIntegerDecrement,
    "XJSupirParameters": XJSupirParameters,
    "XJOneImageFromBatch": OneImageFromBatch,
    "XJLoadImagesFromDirBatch": LoadImagesFromDirBatch,
    "XJLoadImagesFromDirList": LoadImagesFromDirList,
    "XJImageListFilter": XJImageListFilter,
    "XJRandomTextFromList": XJRandomTextFromList,
    "XJRandomTextFromFile": XJRandomTextFromFile,
    "XJRandomImagesFromBatch": XJRandomImagesFromBatch,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSchedulerAdapter": "Scheduler Adapter",
    "XJSamplerAdapter": "Sampler Adapter",
    "XJStringPass": "StringPass",
    "XJImageScaleCalc": "Image Scale Calc",
    "XJImageGrid": "Image Grid",
    "XJIntOffset": "Int Offset",
    "XJImageRandomTransform": "Image Random Transform",
    "XJImageTransform": "Image Transform",
    "XJFloatRangeList": "Float Range List",
    "XJIntegerIncrement": "Integer Increment",
    "XJIntegerDecrement": "Integer Decrement",
    "XJSupirParameters": "SUPIR Parameters",
    "XJOneImageFromBatch": "One Image From Batch",
    "XJLoadImagesFromDirBatch": "Load Images From Dir Batch",
    "XJLoadImagesFromDirList": "Load Images From Dir List",
    "XJImageListFilter": "Image List Filter",
    "XJRandomTextFromList": "Random Text From List",
    "XJRandomTextFromFile": "Random Text From File",
    "XJRandomImagesFromBatch": "Random Images From Batch",
}
