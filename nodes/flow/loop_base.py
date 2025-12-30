"""
Base class for loop implementations in XJNodes

Provides common functionality for loop-like nodes that use subgraph expansion.
Handles graph exploration, reconstruction, and the double-wrapping workaround.
"""

from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_execution.graph import ExecutionBlocker


class LoopBase:
    """
    Base class for implementing loop nodes using ComfyUI's subgraph expansion.

    Subclasses should override:
    - should_exit(): Check if loop should terminate
    - get_loop_values(): Extract values from kwargs
    - get_start_node_id(): Get the ID of the loop start node
    - prepare_start_inputs(): Prepare inputs for next iteration's start node
    """

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        """
        Recursively explore dependencies of a node.

        Traces backward through the graph to find all nodes that feed into this one.

        Args:
            node_id: Current node to explore
            dynprompt: DynamicPrompt containing workflow structure
            upstream: Dict mapping node_id -> list of dependent node IDs
            parent_ids: List to collect parent node IDs
        """
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for key in node_info["inputs"]:
            input_value = node_info["inputs"][key]
            if is_link(input_value):
                parent_id = input_value[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    parent_ids.append(parent_id)
                if node_id not in upstream[parent_id]:
                    upstream[parent_id].append(node_id)
                self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)

    def collect_contained(self, node_id, upstream, contained):
        """
        Collect all nodes contained within the loop body.

        Starting from a node, recursively collect all downstream dependencies.

        Args:
            node_id: Starting node ID
            upstream: Dict mapping node_id -> list of dependent node IDs
            contained: Dict to mark contained nodes (node_id -> True)
        """
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def build_expanded_graph(
        self, dynprompt, unique_id, start_node_id, **iteration_inputs
    ):
        """
        Build the expanded subgraph for the next loop iteration.

        Creates a copy of the loop body with updated inputs for the start node.

        Args:
            dynprompt: DynamicPrompt containing workflow structure
            unique_id: ID of the loop end node
            start_node_id: ID of the loop start node
            **iteration_inputs: Inputs to pass to the start node for next iteration

        Returns:
            dict: {"result": tuple of output links, "expand": finalized graph}
        """
        # Find all nodes in the loop body
        upstream = {}
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)

        # Collect all contained nodes
        contained = {}
        self.collect_contained(start_node_id, upstream, contained)
        contained[unique_id] = True
        contained[start_node_id] = True

        # Build the graph
        graph = GraphBuilder()

        # Create graph nodes for all contained nodes
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            # Use "Recurse" as ID for loop end node to mark recursion point
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

        # Update loop start node with iteration inputs
        new_start = graph.lookup_node(start_node_id)
        for key, value in iteration_inputs.items():
            new_start.set_input(key, value)

        # Get outputs from the recursed loop end node
        my_clone = graph.lookup_node("Recurse")
        # Number of outputs depends on subclass implementation
        # Return links to all outputs
        result = self.get_recursed_outputs(my_clone)

        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }

    def double_wrap_values(self, values):
        """
        Apply double-wrapping workaround for ComfyUI link resolution bug.

        When OUTPUT_IS_LIST=True and passing through subgraph expansion,
        values must be double-wrapped to survive link resolution's unwrapping.

        Args:
            values: List of values to wrap

        Returns:
            List of double-wrapped values: [value] → [[value]]
        """
        return [[v] if v else [] for v in values]

    def unwrap_input_is_list(self, value):
        """
        Unwrap a value from INPUT_IS_LIST wrapping.

        Args:
            value: Value that may be wrapped as [value] from INPUT_IS_LIST

        Returns:
            Unwrapped value, or original if not wrapped
        """
        if isinstance(value, list) and len(value) > 0:
            return value[0]
        return value

    # Abstract methods to be implemented by subclasses

    def should_exit(self, **kwargs):
        """
        Determine if the loop should exit.

        Args:
            **kwargs: Loop state and inputs

        Returns:
            bool: True if loop should exit, False to continue

        Raises:
            NotImplementedError: Subclass must implement
        """
        raise NotImplementedError("Subclass must implement should_exit()")

    def get_loop_values(self, **kwargs):
        """
        Extract loop values from kwargs.

        Args:
            **kwargs: Input parameters

        Returns:
            list: Values to pass through the loop

        Raises:
            NotImplementedError: Subclass must implement
        """
        raise NotImplementedError("Subclass must implement get_loop_values()")

    def get_start_node_id(self, **kwargs):
        """
        Get the ID of the loop start node.

        Args:
            **kwargs: Input parameters containing flow control info

        Returns:
            str: Node ID of the loop start node

        Raises:
            NotImplementedError: Subclass must implement
        """
        raise NotImplementedError("Subclass must implement get_start_node_id()")

    def prepare_start_inputs(self, **kwargs):
        """
        Prepare inputs for the loop start node in the next iteration.

        Args:
            **kwargs: Current loop state and values

        Returns:
            dict: Input values to set on the start node

        Raises:
            NotImplementedError: Subclass must implement
        """
        raise NotImplementedError("Subclass must implement prepare_start_inputs()")

    def get_recursed_outputs(self, recursed_node):
        """
        Get output links from the recursed loop end node.

        Args:
            recursed_node: The graph node representing the recursed loop end

        Returns:
            list: Links to outputs from the recursed node

        Raises:
            NotImplementedError: Subclass must implement
        """
        raise NotImplementedError("Subclass must implement get_recursed_outputs()")
