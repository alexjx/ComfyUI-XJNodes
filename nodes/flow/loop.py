"""
Loop nodes for XJNodes - Iteration control flow

Implementation based on ComfyUI-Easy-Use loop system but simplified:
- Dynamic value handling (no fixed slot limit)
- Values passed as-is (lists remain lists, not expanded)
- Clean direct implementation without wrapper layers
- Refactored to use LoopBase for reusable loop logic

Architecture:
- LoopBase (loop_base.py): Common graph manipulation logic
- XJLoopEnd: Inherits from LoopBase, implements index-based loop
- Future: ForEach can inherit from LoopBase for iterator-based loops

CRITICAL IMPLEMENTATION DETAIL - Why OUTPUT_IS_LIST=True + Double Wrapping:

Goal: Preserve lists as-is (no auto-expansion) when passing through loops.

ComfyUI's default behavior (OUTPUT_IS_LIST=False):
- Wraps each output as "one element": value → [value]
- Normal consumers auto-unwrap and receive value
- Problem: INPUT_IS_LIST consumers receive the wrapped form [value] (double-wrapped!)

Why we use INPUT_IS_LIST=True + OUTPUT_IS_LIST=True:
- INPUT_IS_LIST=True: Receive lists without auto-expansion
- OUTPUT_IS_LIST=True: "My output IS already a list, don't add wrapper"
  - We return [value], ComfyUI keeps it as [value] (no extra wrapping)
  - INPUT_IS_LIST consumers receive [value] directly (no double-wrapping)

Counter-intuitive naming: OUTPUT_IS_LIST=True doesn't mean "expand output",
it means "output IS already in list form, treat as-is".

The double-wrapping workaround for link resolution bug:
- ComfyUI's link resolution (execution.py:457) unconditionally unwraps via iteration
- Designed for OUTPUT_IS_LIST=False (unwrap the [value] wrapper)
- But also runs for OUTPUT_IS_LIST=True (incorrectly unwraps our actual data!)
- Workaround: return [[value]] so after unwrapping we still have [value]

Without double-wrap (BROKEN):
1. Return ([value],) → extend → cached [value]
2. Link resolution unwraps → value
3. merge_result_data extend(value) → if tensor, iterates first dim ❌

With double-wrap (WORKAROUND):
1. Return ([[value]],) → extend → cached [[value]]
2. Link resolution unwraps → [value] ← bug compensated
3. merge_result_data extend([value]) → [value] ✓

See XJLoopEnd.end_loop() line ~350 for detailed comments.
"""

from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_execution.graph import ExecutionBlocker
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from .loop_base import LoopBase

# Maximum number of loop value slots (20 user-accessible slots: value1-value20)
MAX_FLOW_NUM = 20


# Generic type that accepts anything
class AlwaysEqualProxy(str):
    """Type wildcard that matches any type"""

    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


any_type = AlwaysEqualProxy("*")


class ByPassTypeTuple(tuple):
    """
    Special tuple for return types that allows flexible type checking.
    Always returns first element for type validation.
    """

    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return AlwaysEqualProxy(item)
        return item


class XJLoopStart:
    """
    Loop Start Node - Initializes iteration with total count.

    Values are passed through as-is:
    - Single values stay single
    - Lists stay as lists (not expanded item-by-item)
    - Up to 20 value slots supported (value1..value20)

    Special case:
    - total=0: Loop body is skipped, initial values returned unchanged

    The loop body processes values however it wants.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {f"value{i}": (any_type,) for i in range(1, MAX_FLOW_NUM + 1)},
            "hidden": {
                "index": (any_type,),  # Hidden index for iteration state
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ByPassTypeTuple(("FLOW_CONTROL", "INT") + (any_type,) * MAX_FLOW_NUM)
    RETURN_NAMES = ByPassTypeTuple(
        ("flow", "index") + tuple(f"value{i}" for i in range(1, MAX_FLOW_NUM + 1))
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False) + (
        True,
    ) * MAX_FLOW_NUM  # flow/index are singles, values are lists
    FUNCTION = "start_loop"
    CATEGORY = "XJNodes/Flow"

    def start_loop(self, total, unique_id=None, index=None, **kwargs):
        """
        Start or continue loop iteration.

        Args:
            total: Total number of iterations as list [N] (0 = skip loop, return initial values)
            unique_id: Node ID as list [id] (injected by ComfyUI)
            index: Current iteration index as list [i] (0-based, None on first call)
            **kwargs: value1..value20 - each is passed as-is (lists stay lists, singles stay singles)

        Returns:
            Tuple of (flow_control, index, value1, value2, ...)
            If total=0, returns ExecutionBlocker for values to skip loop body
        """
        # With INPUT_IS_LIST = True, all inputs come as lists
        # Extract scalar values from lists
        total = total[0] if isinstance(total, list) else total
        unique_id = unique_id[0] if isinstance(unique_id, list) else unique_id

        # Cast to int (might be string from JSON serialization)
        total = int(total)

        # Initialize index on first call
        if index is None or (isinstance(index, list) and len(index) == 0):
            index = 0
        elif isinstance(index, list):
            index = index[0]

        # Cast index to int
        index = int(index) if index is not None else 0

        # Extract values from kwargs
        # With OUTPUT_IS_LIST=True, None values should be [] instead
        if total == 0:
            # Block loop body execution, return initial values at loop end
            values = [ExecutionBlocker(None) for _ in range(1, MAX_FLOW_NUM + 1)]
        else:
            values = []
            for i in range(1, MAX_FLOW_NUM + 1):
                v = kwargs.get(f"value{i}")
                # Convert None to [] for OUTPUT_IS_LIST compatibility
                values.append([] if v is None else v)

        # Return values as-is (single wrapping only)
        # Loop Start outputs go directly to loop body, NOT through subgraph link resolution
        # So no double-wrapping needed here (only Loop End exit needs it)
        # OUTPUT_IS_LIST=True means "output IS a list, don't add wrapper":
        # We return [value] → ComfyUI keeps as [value] (via extend, no extra wrapping)
        return tuple([(unique_id, total), index] + values)


class XJLoopEnd(LoopBase):
    """
    Loop End Node - Increments counter and continues or exits loop.

    Reconstructs the loop body and expands the graph for next iteration,
    or returns final values if loop is complete.

    Inherits from LoopBase for common loop graph manipulation logic.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": (
                    "FLOW_CONTROL",
                ),  # No rawLink - get evaluated (unique_id, total)
            },
            "optional": {f"value{i}": (any_type,) for i in range(1, MAX_FLOW_NUM + 1)},
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ByPassTypeTuple((any_type,) * MAX_FLOW_NUM)
    RETURN_NAMES = ByPassTypeTuple(
        tuple(f"value{i}" for i in range(1, MAX_FLOW_NUM + 1))
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,) * MAX_FLOW_NUM  # All values are lists
    FUNCTION = "end_loop"
    CATEGORY = "XJNodes/Flow"

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        """
        Recursively explore all nodes that this node depends on.

        Builds a dependency graph from node_id back to all parent nodes,
        stopping at loop start/end nodes.

        Args:
            node_id: Current node to explore
            dynprompt: Dynamic prompt with node graph
            upstream: Dict mapping parent_id -> [child_ids]
            parent_ids: List of all parent node display IDs found
        """
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]

                # Don't traverse through other loop end nodes
                if class_type not in ["XJLoopEnd"]:
                    parent_ids.append(display_id)

                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(
                        parent_id, dynprompt, upstream, parent_ids
                    )

                upstream[parent_id].append(node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        """
        Find output nodes (PreviewImage, SaveImage, etc.) in the loop body.

        Output nodes need special handling to ensure they execute each iteration.
        """
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id in output_nodes:
                id = output_nodes[output_id][0]
                if (
                    id in parent_ids
                    and display_id == id
                    and output_id not in upstream[parent_id]
                ):
                    if "." in parent_id:
                        # Handle ephemeral node IDs (with prefix)
                        arr = parent_id.split(".")
                        arr[len(arr) - 1] = output_id
                        upstream[parent_id].append(".".join(arr))
                    else:
                        upstream[parent_id].append(output_id)

    # collect_contained is inherited from LoopBase

    def end_loop(self, flow, dynprompt=None, unique_id=None, **kwargs):
        """
        End loop iteration - continue or exit.

        Args:
            flow: Evaluated tuple (start_node_id, total) from loop start as list [(node_id, total)]
            dynprompt: Dynamic prompt with execution graph as list [dynprompt]
            unique_id: This node's ID as list [id]
            **kwargs: value1..value20 from loop body (passed as-is)

        Returns:
            If continuing: dict with "result" and "expand" for next iteration
            If done: tuple of final values
        """
        # With INPUT_IS_LIST = True, all inputs come as lists
        # Unwrap using base class helper
        flow = self.unwrap_input_is_list(flow)
        dynprompt = self.unwrap_input_is_list(dynprompt)
        unique_id = self.unwrap_input_is_list(unique_id)

        # flow is the evaluated tuple (start_node_id, total)
        # total is already evaluated, even if it was originally linked to another node
        start_node_id, total = flow

        # Cast total to int (might be from various sources)
        total = int(total)

        # Get start node to access current index
        start_node = dynprompt.get_node(start_node_id)
        current_index = start_node["inputs"].get("index", 0)
        if isinstance(current_index, list):
            current_index = current_index[0] if len(current_index) > 0 else 0
        current_index = int(current_index) if current_index is not None else 0
        next_index = current_index + 1

        # Check if we should exit loop
        if next_index >= total or total == 0:
            # Loop complete - return final values
            # With OUTPUT_IS_LIST=True, None values should be [] instead
            values = []
            for i in range(1, MAX_FLOW_NUM + 1):
                v = kwargs.get(f"value{i}")
                # Convert None to [] for OUTPUT_IS_LIST compatibility
                values.append([] if v is None else v)

            # CRITICAL WORKAROUND: Double-wrap to compensate for ComfyUI link resolution bug
            # See loop_base.py and docs/loop-system.md for detailed explanation
            wrapped_values = self.double_wrap_values(values)

            return tuple(wrapped_values)

        # Continue looping - reconstruct loop body for next iteration
        # Build dependency graph of all nodes in loop body
        upstream = {}
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        # Find output nodes (SaveImage, PreviewImage, etc.) in loop
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for id in prompts:
            node = prompts[id]
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS.get(class_type)
            if (
                class_def
                and hasattr(class_def, "OUTPUT_NODE")
                and class_def.OUTPUT_NODE == True
            ):
                for k, v in node["inputs"].items():
                    if is_link(v):
                        output_nodes[id] = v

        # Build new graph for next iteration
        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)

        # Collect all nodes contained in the loop (from start to end)
        contained = {}
        self.collect_contained(start_node_id, upstream, contained)
        contained[unique_id] = True
        contained[start_node_id] = True

        # Create graph nodes for all contained nodes
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            # Use "Recurse" as ID for this node to mark recursion point
            node = graph.node(
                original_node["class_type"],
                "Recurse" if node_id == unique_id else node_id,
            )
            node.set_override_display_id(node_id)

        # Wire up inputs for all nodes
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    # Internal link - connect to parent node in graph
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    # External input or constant - pass through
                    node.set_input(k, v)

        # Update loop start node with next iteration values
        new_start = graph.lookup_node(start_node_id)
        new_start.set_input("index", next_index)
        for i in range(1, MAX_FLOW_NUM + 1):
            key = f"value{i}"
            # Unwrap INPUT_IS_LIST wrapper before feeding back to Loop Start
            # Loop body outputs value → Loop End receives [value] (INPUT_IS_LIST wrap)
            # Extract value and pass to Loop Start → Loop Start receives [value] again
            # Loop Start passes through to body as [value] (no double-wrap needed here)
            val = kwargs.get(key)
            unwrapped_val = self.unwrap_input_is_list(val) if val is not None else None
            new_start.set_input(key, unwrapped_val)

        # Get outputs from the recursed loop end node
        my_clone = graph.lookup_node("Recurse")
        result = [my_clone.out(i) for i in range(MAX_FLOW_NUM)]

        # IMPORTANT: Return links as-is, no wrapping!
        # These links point to the recursed Loop End node.
        # When the recursed node eventually exits, it will return ([[value]],)
        # Link resolution flow:
        # 1. Recursed node returns ([[value]],) with OUTPUT_IS_LIST=True
        # 2. ComfyUI keeps as [[value]] (no wrapper added)
        # 3. Link resolution unwraps: for o in [[value]] → [value]
        # 4. merge_result_data: extend([value]) → [value] ✓
        # Links must stay as ['node_id', index] format for ComfyUI to recognize them

        expanded_graph = graph.finalize()

        return {
            "result": tuple(
                result
            ),  # Links resolve to double-wrapped values from recursed node
            "expand": expanded_graph,
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "XJLoopStart": XJLoopStart,
    "XJLoopEnd": XJLoopEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJLoopStart": "Loop Start (XJ)",
    "XJLoopEnd": "Loop End (XJ)",
}
