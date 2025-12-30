"""
Flow Control Nodes for XJNodes

Provides control flow primitives for ComfyUI workflows:
- Loop nodes: Iteration control with dynamic value passing
"""

from .loop import NODE_CLASS_MAPPINGS as LOOP_MAPPINGS
from .loop import NODE_DISPLAY_NAME_MAPPINGS as LOOP_DISPLAY_MAPPINGS

# Aggregate all flow control node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(LOOP_MAPPINGS)

# Aggregate all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(LOOP_DISPLAY_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
