"""Conditional Lora Loader - Dynamic lora loading with boolean condition"""

import logging
import folder_paths
from nodes import LoraLoaderModelOnly
from .utils import FlexibleOptionalInputType, any_type

logger = logging.getLogger(__name__)


class XJConditionalLoraLoader:
    """Load multiple loras dynamically with a single boolean condition to enable/disable all"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(type=any_type, data={
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
            }),
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "XJNodes/lora"

    def load_loras(self, model=None, enabled=True, **kwargs):
        """Loops over the provided loras in kwargs and applies them if enabled=True."""
        # If not enabled, just return the model unchanged
        if not enabled or model is None:
            return (model,)

        for key, value in kwargs.items():
            key_upper = key.upper()

            # Look for lora inputs: lora_1, lora_2, etc.
            if key_upper.startswith('LORA_') and isinstance(value, dict):
                # Extract values with defaults
                lora_name = value.get('lora', None)
                strength = value.get('strength', 1.0)

                # Apply lora if valid
                if lora_name and lora_name != 'None' and strength != 0:
                    # Validate lora file exists
                    lora_list = folder_paths.get_filename_list("loras")
                    if lora_name not in lora_list:
                        logger.warning(f"Lora '{lora_name}' not found, skipping")
                        continue

                    # Load the lora (model only, using official LoraLoaderModelOnly)
                    model = LoraLoaderModelOnly().load_lora_model_only(
                        model, lora_name, strength
                    )

        return (model,)


NODE_CLASS_MAPPINGS = {
    "XJConditionalLoraLoader": XJConditionalLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJConditionalLoraLoader": "Conditional Lora Loader",
}
