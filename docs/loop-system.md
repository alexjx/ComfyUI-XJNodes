# Loop System Implementation

This document explains how the XJNodes loop system works, including the technical details and workarounds required for ComfyUI compatibility.

## Overview

The loop system provides iteration control flow using `XJLoopStart` and `XJLoopEnd` nodes. It's implemented using ComfyUI's subgraph expansion mechanism to execute the loop body multiple times.

## Core Concepts

### ComfyUI's Default Behavior (OUTPUT_IS_LIST=False)

By default, ComfyUI wraps each node output as a "single element":
```python
value → [value]  # ComfyUI adds wrapper
```

Normal consumers auto-unwrap this. However, nodes with `INPUT_IS_LIST=True` receive the wrapped form directly:
```python
# Normal node receives: value (auto-unwrapped)
# INPUT_IS_LIST node receives: [value] (wrapped form)
```

**Problem**: If `value` is already a list `[item1, item2]`, it becomes double-wrapped `[[item1, item2]]` for `INPUT_IS_LIST` consumers.

### Why We Use INPUT_IS_LIST=True + OUTPUT_IS_LIST=True

**Goal**: Preserve lists as-is without auto-expansion.

**INPUT_IS_LIST=True**: Receive all inputs as lists without ComfyUI auto-expanding them.

**OUTPUT_IS_LIST=True**: Tell ComfyUI "my output IS already in list form, don't add wrapper"
- We return `[value]`, ComfyUI keeps it as `[value]` (no extra wrapping)
- `INPUT_IS_LIST` consumers receive `[value]` directly (no double-wrapping)

**Counter-intuitive naming**: OUTPUT_IS_LIST=True doesn't mean "expand output", it means "output IS already a list, treat as-is".

### The Link Resolution Bug

ComfyUI's subgraph link resolution (execution.py:457-458) has a bug:

```python
# execution.py:457-458
for o in cached.outputs[i]:
    resolved_output.append(o)
```

This unconditionally unwraps by iterating, designed for `OUTPUT_IS_LIST=False` (to unwrap the `[value]` wrapper), but it also runs for `OUTPUT_IS_LIST=True` (incorrectly unwrapping our actual data).

**Impact**: When loop outputs pass through subgraph expansion, they get unwrapped one extra time, causing IMAGE batch dimensions to be lost.

### The Double-Wrapping Workaround

To compensate for the link resolution bug, we double-wrap outputs from `XJLoopEnd`:

**Without double-wrap (BROKEN)**:
1. Loop End returns `([value],)` with OUTPUT_IS_LIST=True
2. ComfyUI keeps as `[value]` (no wrapper added, as intended)
3. Link resolution unwraps: `for o in [value]` → `value`
4. Second merge_result_data: `extend(value)` → if tensor, iterates first dimension ❌

**With double-wrap (WORKAROUND)**:
1. Loop End returns `([[value]],)` with OUTPUT_IS_LIST=True
2. ComfyUI keeps as `[[value]]` (no wrapper added)
3. Link resolution unwraps: `for o in [[value]]` → `[value]` ← bug compensated
4. Second merge_result_data: `extend([value])` → `[value]` preserved ✓

## Loop Flow

### Initialization (XJLoopStart)

```
User connects values to XJLoopStart
  ↓
XJLoopStart receives values (INPUT_IS_LIST wrapping)
  ↓
If total > 0:
  - Returns (flow_control, index=0, values...)
  - Loop body executes with initial values
If total = 0:
  - Returns ExecutionBlocker to skip loop body
  - XJLoopEnd returns initial values immediately
```

### Loop Iteration (XJLoopEnd - Continue)

```
Loop body outputs value
  ↓
XJLoopEnd receives [value] (INPUT_IS_LIST wrapping)
  ↓
Check exit condition: next_index >= total?
  ↓ NO (continue looping)
Build expanded subgraph:
  1. Find all nodes in loop body (between Start and End)
  2. Reconstruct graph with same connections
  3. Update Loop Start inputs with current iteration values
  4. Return links to recursed Loop End outputs
  ↓
ComfyUI executes expanded graph
  ↓
Recursed Loop End eventually exits or continues again
```

### Loop Exit (XJLoopEnd - Exit)

```
Check exit condition: next_index >= total?
  ↓ YES (exit loop)
Extract values from kwargs
  ↓
Double-wrap values: [value] → [[value]]
  (Workaround for link resolution bug)
  ↓
Return ([[value]],) with OUTPUT_IS_LIST=True
  ↓
Link resolution unwraps: [[value]] → [value]
  ↓
Consumer receives [value] correctly
```

## Graph Expansion Details

### Finding Loop Body Nodes

1. **Explore dependencies**: Trace backward from Loop End to find all nodes that feed into it
2. **Collect contained nodes**: Get all nodes between Loop Start and Loop End
3. **Filter**: Only include nodes within the loop boundary

### Reconstructing the Graph

```python
# Create nodes in expanded graph
for node_id in contained:
    original_node = dynprompt.get_node(node_id)
    graph.node(original_node["class_type"], node_id)

# Wire up connections
for node_id in contained:
    original_node = dynprompt.get_node(node_id)
    for input_name, input_value in original_node["inputs"].items():
        if is_link(input_value) and input_value[0] in contained:
            # Internal link - connect to parent in graph
            parent = graph.lookup_node(input_value[0])
            node.set_input(input_name, parent.out(input_value[1]))
        else:
            # External input - pass through
            node.set_input(input_name, input_value)
```

### Updating Loop Start for Next Iteration

The Loop Start node in the expanded graph gets updated values:
```python
new_start = graph.lookup_node(start_node_id)
new_start.set_input("index", next_index)
for i in range(1, MAX_FLOW_NUM + 1):
    # Unwrap INPUT_IS_LIST wrapper
    val = kwargs.get(f"value{i}")
    if val and isinstance(val, list) and len(val) > 0:
        new_start.set_input(f"value{i}", val[0])
```

## Data Flow Example

```
Iteration 0:
  Loop Start → [initial_value] → Loop Body → [result_0] → Loop End
  Loop End continues, builds expanded graph pointing to itself

Iteration 1:
  Recursed Loop Start → [result_0] → Loop Body → [result_1] → Recursed Loop End
  Recursed Loop End continues, builds another expanded graph

Iteration 2:
  Recursed Loop Start → [result_1] → Loop Body → [result_2] → Recursed Loop End
  Recursed Loop End exits (next_index >= total)

Exit:
  Recursed Loop End returns ([[result_2]],)
  Link resolution: [[result_2]] → [result_2]
  Final consumer receives: [result_2]
```

## Common Patterns

### Using with Image Lists

```
XJEmptyImageList → Loop Start (value1)
                        ↓
                    Load Image
                        ↓
                 XJAppendImageList (connects to value1)
                        ↓
                    Loop End (value1)
                        ↓
                 XJUnwrapFromList
                        ↓
                  Image Preview
```

The `XJUnwrapFromList` node handles the double-wrapping:
- Receives `[[image_list]]` from Loop End
- Unwraps twice → `image_list`
- Returns to normal consumers

### Handling None Values

Both Loop Start and Loop End convert `None` to `[]` for OUTPUT_IS_LIST compatibility:
```python
values.append([] if v is None else v)
```

This prevents ComfyUI from treating `None` differently than empty lists.

## Implementation Notes

### Why Not Use Easy-Use Pattern?

Easy-Use loop nodes use neither `INPUT_IS_LIST` nor `OUTPUT_IS_LIST`. They pass values directly, which works for simple cases but:
- Lists get auto-expanded by ComfyUI
- Cannot preserve list structure through loops
- Requires manual wrapping/unwrapping by users

Our implementation prioritizes "preserve lists as-is" which requires the INPUT_IS_LIST + OUTPUT_IS_LIST pattern.

### Trade-offs

**Pros**:
- Lists preserved without auto-expansion
- Batch dimensions maintained correctly
- Consistent handling with other list nodes

**Cons**:
- Requires double-wrapping workaround
- More complex implementation
- Needs XJUnwrapFromList for normal consumers

### Future Improvements

If ComfyUI fixes the link resolution bug to respect `OUTPUT_IS_LIST`:
1. Remove double-wrapping in Loop End exit
2. Simplify XJUnwrapFromList (only single unwrap needed)
3. Update documentation

The workaround is isolated to Loop End's exit path and can be easily removed when no longer needed.

## References

- ComfyUI execution.py:457-458 (link resolution)
- ComfyUI execution.py:323 (OUTPUT_IS_LIST=True extend)
- ComfyUI execution.py:326 (OUTPUT_IS_LIST=False wrap)
- ComfyUI docs: https://docs.comfy.org/custom-nodes/backend/lists
