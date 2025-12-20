import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import hashlib
import nodes
import logging

# ExecutionBlocker exists but doesn't reliably stop workflow in all contexts
# Using exception-based approach for reliable stopping
HAS_EXECUTION_BLOCKER = False

# Global cache for tracking image state per node instance
# Key: unique_id, Value: (image_hash, preview_filename)
_preview_bridge_cache = {}


class XJImagePreviewBridge:
    """
    A bridge node that accepts images from any source, allows interactive mask editing,
    and passes through both image and mask to downstream nodes.

    Features:
    - Accepts IMAGE input from any node (generation, upscaling, etc.)
    - Saves image to temp and shows preview in node
    - Right-click "Open with Mask Editor" to create mask interactively
    - Outputs passthrough IMAGE and created MASK
    - Optional workflow stop if mask is empty
    - Persistent storage using widgets for image/mask filenames
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # IMAGE tensor input
                "stop_if_empty": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Visible STRING widget named "image" - for mask editor and debugging
                # Mask editor automatically updates this with mask filename
                # Shows the clipspace filename for debugging
                "image": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "XJNodes/image"
    OUTPUT_NODE = True

    @staticmethod
    def validate_mask_for_image(mask_file_path, current_image_tensor):
        """
        Validate mask by comparing non-masked areas with current image.

        The mask editor saves a composite image with:
        - RGB channels: The original image when mask was created
        - Alpha channel: The mask

        We compare non-masked areas to verify the mask belongs to current image.

        Args:
            mask_file_path: Path to clipspace composite file
            current_image_tensor: Current input [B, H, W, C] in [0, 1]

        Returns:
            (is_valid: bool, reason: str, mask_tensor: Tensor or None)
        """
        try:
            # Load mask file (composite)
            mask_img = Image.open(mask_file_path)

            if 'A' not in mask_img.getbands():
                return False, "No alpha channel in mask file", None

            # Extract components
            rgb_from_file = np.array(mask_img.convert('RGB')).astype(np.float32) / 255.0
            alpha = np.array(mask_img.getchannel('A')).astype(np.float32) / 255.0

            # Current image (first in batch)
            current_img = current_image_tensor[0].cpu().numpy()  # [H, W, C]

            # Check dimensions
            if rgb_from_file.shape != current_img.shape:
                return False, f"Size mismatch: mask {rgb_from_file.shape} vs image {current_img.shape}", None

            # Create mask tensor (ComfyUI convention: inverted)
            mask_tensor = torch.from_numpy(1.0 - alpha).unsqueeze(0)

            # Check not empty
            if mask_tensor.sum() == 0:
                return False, "Mask is empty (all zeros)", None

            # Find non-masked pixels (where mask == 0, meaning not selected)
            non_masked = (mask_tensor[0] == 0)  # [H, W] boolean

            if non_masked.sum() == 0:
                # Entire image is masked - can't validate, but accept
                return True, "Valid (entire image masked, can't verify)", mask_tensor

            # Compare non-masked areas
            diff = np.abs(rgb_from_file - current_img)
            non_masked_diff = diff[non_masked.numpy()]

            max_diff = non_masked_diff.max()
            mean_diff = non_masked_diff.mean()

            # Tolerance for JPEG compression artifacts
            tolerance = 0.02  # 2% = ~5 levels in 0-255 range

            if max_diff > tolerance:
                return False, f"Image content mismatch (max_diff={max_diff:.3f}, mean={mean_diff:.3f})", None

            return True, f"Valid (max_diff={max_diff:.4f}, mean={mean_diff:.4f})", mask_tensor

        except Exception as e:
            return False, f"Error loading mask: {str(e)}", None

    def execute(self, images, stop_if_empty=True, image="", unique_id=None, prompt=None, extra_pnginfo=None):
        """
        Execute the preview bridge with proper state handling.

        States:
        1. Image changed/first run: Save preview, check mask, optionally stop
        2. Image unchanged, no mask: Reuse preview, optionally stop
        3. Image unchanged, mask exists: Validate mask, continue if valid

        Args:
            images: Input IMAGE tensor
            stop_if_empty: Whether to stop workflow if mask is empty/invalid
            image: Widget string value (mask filename set by mask editor)
            unique_id: Node's unique ID
        """

        # Calculate hash of current images to detect changes
        current_image_hash = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()

        # Check global cache for this node's previous state
        cached_state = _preview_bridge_cache.get(unique_id)
        cached_hash, cached_filename = cached_state if cached_state else (None, None)

        # Determine if image has changed
        image_changed = (cached_hash != current_image_hash) or (cached_hash is None)

        # ========== STATE 1: Image Changed or First Run ==========
        if image_changed:
            logging.info(f"[XJNodes] PreviewBridge {unique_id}: Image changed, saving new preview")

            # Save NEW preview image
            filename_prefix = f"PreviewBridge/{unique_id}_" if unique_id else "PreviewBridge/default_"
            preview_result = nodes.PreviewImage().save_images(
                images,
                filename_prefix=filename_prefix,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo
            )

            saved_images = preview_result['ui']['images']
            if not saved_images:
                raise RuntimeError("Failed to save preview image")

            saved_image_info = saved_images[0]
            saved_filename = saved_image_info['filename']
            saved_subfolder = saved_image_info.get('subfolder', '')
            saved_type = saved_image_info.get('type', 'temp')

            # Update global cache
            _preview_bridge_cache[unique_id] = (current_image_hash, saved_filename)

            # Check if mask widget has value (might be from old image)
            if image:
                # Parse mask file path with type
                # Format: "subfolder/filename [type]" or "filename [type]"
                # Always respect the [type] suffix - don't assume any specific type
                image_value = image.strip()
                mask_type = "input"  # Default if no suffix

                if '[' in image_value and ']' in image_value:
                    type_start = image_value.rfind('[')
                    type_end = image_value.rfind(']')
                    mask_type = image_value[type_start+1:type_end].strip()
                    image_value = image_value[:type_start].strip()

                if '/' in image_value:
                    parts = image_value.rsplit('/', 1)
                    mask_subfolder = parts[0]
                    mask_filename = parts[1]
                else:
                    mask_subfolder = 'clipspace' if mask_type == 'input' else ''
                    mask_filename = image_value

                # Get appropriate folder based on [type] suffix
                if mask_type == "temp":
                    base_dir = folder_paths.get_temp_directory()
                elif mask_type == "output":
                    base_dir = folder_paths.get_output_directory()
                else:  # "input" or any other type
                    base_dir = folder_paths.get_input_directory()

                mask_path = os.path.join(base_dir, mask_subfolder, mask_filename) if mask_subfolder else os.path.join(base_dir, mask_filename)

                # Validate mask belongs to NEW image
                is_valid, reason, validated_mask = self.validate_mask_for_image(mask_path, images)

                if is_valid:
                    logging.info(f"[XJNodes] PreviewBridge {unique_id}: Mask valid for new image: {reason}")
                    output_mask = validated_mask
                    should_stop = False
                else:
                    logging.warning(f"[XJNodes] PreviewBridge {unique_id}: Mask invalid for new image: {reason}")
                    # Widget says mask exists but we failed to load it - always treat as error
                    batch, height, width, _ = images.shape
                    output_mask = torch.zeros((batch, height, width), dtype=torch.float32)
                    should_stop = True  # Always stop if referenced mask can't be loaded
                    stop_reason = f"Image changed. Old mask invalid: {reason}\nPlease create a new mask."
            else:
                # No mask widget value
                batch, height, width, _ = images.shape
                output_mask = torch.zeros((batch, height, width), dtype=torch.float32)
                should_stop = stop_if_empty
                stop_reason = "Preview shown. Please create a mask:\n1. Right-click the node\n2. Select 'Open with Mask Editor'\n3. Create your mask\n4. Save and re-run"

        # ========== STATE 2 & 3: Image Unchanged ==========
        else:
            logging.info(f"[XJNodes] PreviewBridge {unique_id}: Image unchanged, reusing preview")

            # Reuse cached preview - NO need to save again
            saved_filename = cached_filename
            saved_subfolder = 'PreviewBridge'
            saved_type = 'temp'
            saved_images = [{
                'filename': saved_filename,
                'subfolder': saved_subfolder,
                'type': saved_type
            }]

            # Check mask widget
            if not image:
                # STATE 2: No mask widget value
                logging.info(f"[XJNodes] PreviewBridge {unique_id}: No mask created yet")
                batch, height, width, _ = images.shape
                output_mask = torch.zeros((batch, height, width), dtype=torch.float32)
                should_stop = stop_if_empty
                stop_reason = "No mask created yet.\nPlease create a mask via 'Open with Mask Editor'."
            else:
                # STATE 3: Mask exists - validate it
                # Parse mask file path with type
                # Format: "subfolder/filename [type]" or "filename [type]"
                # Always respect the [type] suffix - don't assume any specific type
                image_value = image.strip()
                mask_type = "input"  # Default if no suffix

                if '[' in image_value and ']' in image_value:
                    type_start = image_value.rfind('[')
                    type_end = image_value.rfind(']')
                    mask_type = image_value[type_start+1:type_end].strip()
                    image_value = image_value[:type_start].strip()

                if '/' in image_value:
                    parts = image_value.rsplit('/', 1)
                    mask_subfolder = parts[0]
                    mask_filename = parts[1]
                else:
                    mask_subfolder = 'clipspace' if mask_type == 'input' else ''
                    mask_filename = image_value

                # Get appropriate folder based on [type] suffix
                if mask_type == "temp":
                    base_dir = folder_paths.get_temp_directory()
                elif mask_type == "output":
                    base_dir = folder_paths.get_output_directory()
                else:  # "input" or any other type
                    base_dir = folder_paths.get_input_directory()

                mask_path = os.path.join(base_dir, mask_subfolder, mask_filename) if mask_subfolder else os.path.join(base_dir, mask_filename)

                # Validate mask for current image
                is_valid, reason, validated_mask = self.validate_mask_for_image(mask_path, images)

                if is_valid:
                    logging.info(f"[XJNodes] PreviewBridge {unique_id}: Mask valid: {reason}")
                    output_mask = validated_mask
                    should_stop = False
                else:
                    logging.error(f"[XJNodes] PreviewBridge {unique_id}: Mask invalid: {reason}")
                    # Widget says mask exists but we failed to load it - always treat as error
                    batch, height, width, _ = images.shape
                    output_mask = torch.zeros((batch, height, width), dtype=torch.float32)
                    should_stop = True  # Always stop if referenced mask can't be loaded
                    stop_reason = f"Mask validation failed: {reason}\nPlease create a new mask."

        # ========== Execute or Block Based on State ==========
        if should_stop:
            if HAS_EXECUTION_BLOCKER:
                logging.info(f"[XJNodes] PreviewBridge: Stopping workflow - {stop_reason}")
                result = (ExecutionBlocker(None), ExecutionBlocker(None))
            else:
                raise ValueError(f"⚠️ Workflow stopped:\n{stop_reason}")
        else:
            # Continue with actual data
            result = (images, output_mask)

        return {
            "ui": {"images": saved_images},
            "result": result
        }

    @classmethod
    def IS_CHANGED(cls, stop_if_empty=True, unique_id=None, image="", **kwargs):
        """
        Smart cache invalidation:
        - Returns different cache keys based on mask file existence
        - When mask created/modified, cache is invalidated
        - When mask unchanged, cache is preserved for performance

        NOTE: IS_CHANGED receives widget values only, NOT tensor data!
        We check filesystem state instead.
        """
        if not unique_id:
            # No unique_id means we can't track state properly
            return float("NaN")  # Always re-execute

        # Check if "image" widget has a value (updated by mask editor)
        if not image:
            # No image widget value - first run or mask not created
            return f"no_mask_init_{unique_id}"

        # Parse the "image" widget value to find mask file
        # Format: "subfolder/filename [type]" or "filename [type]"
        # Always respect the [type] suffix - don't assume any specific type
        image_value = image.strip()
        mask_type = "input"  # Default if no suffix

        # Extract [type] suffix
        if '[' in image_value and ']' in image_value:
            type_start = image_value.rfind('[')
            type_end = image_value.rfind(']')
            mask_type = image_value[type_start+1:type_end].strip()
            image_value = image_value[:type_start].strip()

        # Split by '/' to get subfolder and filename
        if '/' in image_value:
            parts = image_value.rsplit('/', 1)
            mask_subfolder = parts[0]
            mask_filename = parts[1]
        else:
            mask_subfolder = 'clipspace' if mask_type == 'input' else ''
            mask_filename = image_value

        # Get appropriate folder based on [type] suffix
        if mask_type == "temp":
            base_dir = folder_paths.get_temp_directory()
        elif mask_type == "output":
            base_dir = folder_paths.get_output_directory()
        else:  # "input" or any other type
            base_dir = folder_paths.get_input_directory()

        mask_path = os.path.join(base_dir, mask_subfolder, mask_filename) if mask_subfolder else os.path.join(base_dir, mask_filename)

        # Check if mask file exists and hash it
        if os.path.exists(mask_path):
            try:
                with open(mask_path, 'rb') as f:
                    mask_hash = hashlib.md5(f.read()).hexdigest()
                return f"has_mask_{mask_hash[:16]}"
            except OSError:
                pass

        # No mask exists yet
        return f"no_mask_{unique_id}"


NODE_CLASS_MAPPINGS = {
    "XJImagePreviewBridge": XJImagePreviewBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJImagePreviewBridge": "Image Preview Bridge",
}
