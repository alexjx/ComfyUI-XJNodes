import random
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageEnhance, ImageOps


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
    CATEGORY = "XJNodes/image"

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
    CATEGORY = "XJNodes/image"

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


NODE_CLASS_MAPPINGS = {
    "XJImageRandomTransform": XJImageRandomTransform,
    "XJImageTransform": XJImageTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJImageRandomTransform": "Image Random Transform",
    "XJImageTransform": "Image Transform",
}
