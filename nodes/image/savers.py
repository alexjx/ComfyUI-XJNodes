import os
import time
import re
import socket
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import comfy.model_management
import folder_paths


def parse_filename_tokens(text):
    """Parse filename tokens similar to WAS nodes"""
    # Basic tokens
    tokens = {
        "[time]": str(int(time.time())),
        "[hostname]": socket.gethostname(),
    }

    # Try to get username
    try:
        tokens["[user]"] = os.getlogin() if os.getlogin() else "user"
    except Exception:
        tokens["[user]"] = "user"

    # Try to get CUDA info
    try:
        tokens["[cuda_device]"] = str(comfy.model_management.get_torch_device())
        tokens["[cuda_name]"] = str(
            comfy.model_management.get_torch_device_name(
                device=comfy.model_management.get_torch_device()
            )
        )
    except Exception:
        tokens["[cuda_device]"] = "cpu"
        tokens["[cuda_name]"] = "unknown"

    # Replace simple tokens
    for token, value in tokens.items():
        text = text.replace(token, value)

    # Handle [time(%Y-%m-%d)] format tokens
    def replace_time_format(match):
        format_code = match.group(1)
        return time.strftime(format_code, time.localtime(time.time()))

    text = re.sub(r"\[time\((.*?)\)\]", replace_time_format, text)

    return text


class XJSaveImageWithMetadata:
    """Save image with custom metadata - full featured like WAS Image Save"""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "number_padding": (
                    "INT",
                    {"default": 5, "min": 1, "max": 10, "step": 1},
                ),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_mode": (
                    ["prefix_number", "number_prefix", "number_only"],
                    {"default": "prefix_number"},
                ),
                "extension": (["png", "jpg", "jpeg", "webp", "bmp", "tiff"],),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "dpi": ("INT", {"default": 300, "min": 72, "max": 2400, "step": 1}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "optimize": ("BOOLEAN", {"default": False}),
                "lossless_webp": ("BOOLEAN", {"default": False}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                "metadata": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "XJNode/image"

    def save_images(
        self,
        images,
        output_path="",
        filename_prefix="ComfyUI",
        extension="png",
        quality=95,
        dpi=300,
        optimize=False,
        lossless_webp=False,
        embed_workflow=True,
        metadata="",
        number_padding=5,
        filename_delimiter="_",
        filename_mode="prefix_number",
        overwrite_existing=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        # Parse tokens in filename_prefix and output_path
        filename_prefix = parse_filename_tokens(filename_prefix)
        output_path = parse_filename_tokens(output_path)

        # Setup output path
        if output_path in [None, "", "none", "."]:
            output_folder = self.output_dir
        else:
            if not os.path.isabs(output_path):
                output_folder = os.path.join(self.output_dir, output_path)
            else:
                output_folder = output_path

        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # Get counter for filename
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, output_folder, images[0].shape[1], images[0].shape[0]
            )
        )

        # For "number_only" mode, scan for all numbered files regardless of prefix
        if filename_mode == "number_only":
            # Find the highest numbered file in the directory
            max_counter = 0
            if os.path.exists(full_output_folder):
                for existing_file in os.listdir(full_output_folder):
                    # Match files that start with digits followed by any delimiter
                    match = re.match(r"^(\d+)[\._-]", existing_file)
                    if match:
                        file_num = int(match.group(1))
                        max_counter = max(max_counter, file_num)
            counter = max_counter + 1

        results = []
        file_paths = []

        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Prepare metadata based on format
            if extension == "webp":
                # WebP uses EXIF
                img_exif = img.getexif()
                if embed_workflow:
                    if prompt is not None:
                        img_exif[0x010F] = "Prompt:" + json.dumps(prompt)
                    if extra_pnginfo is not None:
                        workflow_metadata = ""
                        for x in extra_pnginfo:
                            workflow_metadata += json.dumps(extra_pnginfo[x])
                        img_exif[0x010E] = "Workflow:" + workflow_metadata

                # Add custom metadata to EXIF UserComment
                if metadata:
                    img_exif[0x9286] = metadata  # UserComment

                exif_data = img_exif.tobytes()
            else:
                # PNG, TIFF, BMP use PngInfo
                pnginfo = PngInfo()
                if embed_workflow:
                    if prompt is not None:
                        pnginfo.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            pnginfo.add_text(x, json.dumps(extra_pnginfo[x]))

                # Add custom metadata to our dedicated key
                if metadata:
                    pnginfo.add_text("xj_metadata", metadata)

                exif_data = pnginfo

            # Generate filename with custom delimiter and padding
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

            # Format counter with custom padding
            counter_str = str(counter).zfill(number_padding)

            # Build filename based on mode
            if filename_mode == "number_only" or filename_mode == "number_prefix":
                # number_only: 00001_ComfyUI.png (global counter)
                # number_prefix: 00001_ComfyUI.png (prefix-specific counter)
                file = f"{counter_str}{filename_delimiter}{filename_with_batch_num}.{extension}"
            else:  # prefix_number
                # prefix_number: ComfyUI_00001.png
                file = f"{filename_with_batch_num}{filename_delimiter}{counter_str}.{extension}"
            file_path = os.path.join(full_output_folder, file)

            # Anti-overwrite: if file exists and overwrite is disabled, find next available number
            if not overwrite_existing:
                while os.path.exists(file_path):
                    counter += 1
                    counter_str = str(counter).zfill(number_padding)
                    if filename_mode == "number_only" or filename_mode == "number_prefix":
                        file = f"{counter_str}{filename_delimiter}{filename_with_batch_num}.{extension}"
                    else:  # prefix_number
                        file = f"{filename_with_batch_num}{filename_delimiter}{counter_str}.{extension}"
                    file_path = os.path.join(full_output_folder, file)

            # Save image based on format
            try:
                if extension in ["jpg", "jpeg"]:
                    # JPEG doesn't support PNG metadata, convert to RGB
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(
                        file_path, quality=quality, optimize=optimize, dpi=(dpi, dpi)
                    )
                elif extension == "webp":
                    img.save(
                        file_path,
                        quality=quality,
                        lossless=lossless_webp,
                        exif=exif_data,
                    )
                elif extension == "png":
                    img.save(
                        file_path, pnginfo=exif_data, optimize=optimize, dpi=(dpi, dpi)
                    )
                elif extension == "bmp":
                    img.save(file_path)
                elif extension == "tiff":
                    img.save(
                        file_path, quality=quality, optimize=optimize, dpi=(dpi, dpi)
                    )
                else:
                    img.save(file_path, pnginfo=exif_data, optimize=optimize)

                file_paths.append(file_path)
                results.append(
                    {"filename": file, "subfolder": subfolder, "type": self.type}
                )
                counter += 1

            except Exception as e:
                print(f"Error saving {file_path}: {e}")

        return {"ui": {"images": results}, "result": (file_paths,)}


NODE_CLASS_MAPPINGS = {
    "XJSaveImageWithMetadata": XJSaveImageWithMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSaveImageWithMetadata": "Save Image With Metadata",
}
