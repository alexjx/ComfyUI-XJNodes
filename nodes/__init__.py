"""
ComfyUI-XJNodes - Modular Node Package

This package contains custom nodes organized by data type:
- image: Image processing nodes (loaders, savers, transforms, scaling, batch operations)
- text: Text processing nodes (loaders, info)
- primitives: Basic data type operations (adapters, integers, floats)
- lora: Lora loading nodes (conditional loader)
- specialized: Special-purpose nodes (SUPIR helpers)
"""

from .image import NODE_CLASS_MAPPINGS as IMAGE_MAPPINGS
from .image import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DISPLAY_MAPPINGS
from .text import NODE_CLASS_MAPPINGS as TEXT_MAPPINGS
from .text import NODE_DISPLAY_NAME_MAPPINGS as TEXT_DISPLAY_MAPPINGS
from .primitives import NODE_CLASS_MAPPINGS as PRIMITIVES_MAPPINGS
from .primitives import NODE_DISPLAY_NAME_MAPPINGS as PRIMITIVES_DISPLAY_MAPPINGS
from .lora import NODE_CLASS_MAPPINGS as LORA_MAPPINGS
from .lora import NODE_DISPLAY_NAME_MAPPINGS as LORA_DISPLAY_MAPPINGS
from .specialized import NODE_CLASS_MAPPINGS as SPECIALIZED_MAPPINGS
from .specialized import NODE_DISPLAY_NAME_MAPPINGS as SPECIALIZED_DISPLAY_MAPPINGS

# Aggregate all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(IMAGE_MAPPINGS)
NODE_CLASS_MAPPINGS.update(TEXT_MAPPINGS)
NODE_CLASS_MAPPINGS.update(PRIMITIVES_MAPPINGS)
NODE_CLASS_MAPPINGS.update(LORA_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SPECIALIZED_MAPPINGS)

# Aggregate all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(TEXT_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PRIMITIVES_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LORA_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SPECIALIZED_DISPLAY_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
