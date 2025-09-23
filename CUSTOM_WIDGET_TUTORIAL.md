# ComfyUI Custom Widget Tutorial

A comprehensive guide on how to create custom widgets for ComfyUI nodes, based on successfully implementing a code editor widget with line numbers.

## Overview

Custom widgets in ComfyUI allow you to create sophisticated UI components that go beyond the standard input types. This tutorial covers the complete process from Python node definition to JavaScript widget implementation.

## Architecture

ComfyUI custom widgets work through a connection between Python and JavaScript:

1. **Python Node**: Defines a custom input type
2. **JavaScript Extension**: Registers a widget factory for that type
3. **Widget Factory**: Creates the actual DOM widget
4. **ComfyUI Integration**: Handles serialization, resizing, and lifecycle

## Step 1: Python Node Definition

### File Structure
```
your-extension/
├── __init__.py
├── nodes.py           # Python node definitions
└── web/
    └── js/
        └── node.js    # JavaScript widget implementations
```

### Python Code (`nodes.py`)

```python
class YourCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Key: Use a custom widget type instead of "STRING"
                "your_input": ("YOUR_CUSTOM_WIDGET", {}),
                "other_param": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "your_category"

    def process(self, your_input, other_param):
        # your_input will contain the value from your custom widget
        return (your_input,)

# Register the node
NODE_CLASS_MAPPINGS = {
    "YourCustomNode": YourCustomNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YourCustomNode": "Your Custom Node"
}
```

**Key Points:**
- Use a **custom type name** (`"YOUR_CUSTOM_WIDGET"`) instead of standard types
- This type name **must match** the JavaScript widget registration
- The input parameter will receive the widget's value during execution

## Step 2: JavaScript Widget Extension

### File Structure (`web/js/node.js`)

```javascript
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "YourExtension.CustomWidget",

    // This is where you register your custom widget factory
    getCustomWidgets() {
        return {
            // Widget type name MUST match the Python type
            YOUR_CUSTOM_WIDGET(node, inputName) {
                console.log("Creating widget for:", inputName);

                // Create your DOM elements
                const container = document.createElement("div");
                container.style.cssText = `
                    width: 100%;
                    height: 100%;
                    border: 1px solid #555;
                    background: #2a2a2a;
                    box-sizing: border-box;
                `;

                const input = document.createElement("input");
                input.type = "text";
                input.style.cssText = `
                    width: 100%;
                    height: 100%;
                    border: none;
                    background: transparent;
                    color: #fff;
                    padding: 8px;
                `;

                const defaultValue = "Default text";
                input.value = defaultValue;

                container.appendChild(input);

                // Create the widget using ComfyUI's addDOMWidget
                const widget = node.addDOMWidget(inputName, "your_widget_type", container, {
                    serialize: true,  // Save/load values
                    getValue() {
                        return input.value;
                    },
                    setValue(v) {
                        input.value = v;
                    },
                    getMinHeight() {
                        return 50;  // Minimum height in pixels
                    },
                    getMaxHeight() {
                        return 200; // Maximum height in pixels
                    }
                });

                // Set initial value
                widget.value = defaultValue;

                // Add event listeners
                input.addEventListener('input', () => {
                    // Update widget value when user types
                    widget.value = input.value;
                });

                console.log("Widget created successfully");

                // IMPORTANT: Return the widget wrapped in an object
                return { widget };
            }
        };
    },

    // Optional: Handle node creation for additional setup
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "YourCustomNode") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = originalNodeCreated?.apply(this, arguments);

                // Additional node setup if needed
                console.log("Node created:", this);

                return result;
            };
        }
    }
});
```

## Step 3: Essential Code Points

### 1. Widget Type Connection
```python
# Python
"input_name": ("CUSTOM_TYPE", {})
```
```javascript
// JavaScript - MUST match exactly
getCustomWidgets() {
    return {
        CUSTOM_TYPE(node, inputName) { /* ... */ }
    };
}
```

### 2. DOM Widget Creation
```javascript
const widget = node.addDOMWidget(inputName, "widget_type", domElement, options);
```

**Required options:**
- `serialize: true` - Enables save/load functionality
- `getValue()` - Returns current widget value
- `setValue(v)` - Sets widget value programmatically

**Optional options:**
- `getMinHeight()` - Minimum widget height
- `getMaxHeight()` - Maximum widget height
- `afterResize(node)` - Called when widget is resized

### 3. Return Format
```javascript
// IMPORTANT: Must return widget wrapped in object
return { widget };
```

### 4. Event Handling
```javascript
input.addEventListener('input', () => {
    widget.value = input.value;  // Update widget value
});
```

## Step 4: Advanced Example - Code Editor

Here's a more complex example showing how we implemented the code editor widget:

```javascript
YOUR_CODE_EDITOR(node, inputName) {
    // Create container with flexbox layout
    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        width: 100%;
        height: 100%;
        font-family: 'Courier New', monospace;
        border: 1px solid #555;
        background: #1a1a1a;
    `;

    // Line number gutter
    const gutter = document.createElement('div');
    gutter.style.cssText = `
        width: 50px;
        background: #2a2a2a;
        border-right: 1px solid #555;
        padding: 8px 4px;
        text-align: right;
        color: #888;
        overflow: hidden;
        user-select: none;
        line-height: 18px;
    `;

    // Text editing area
    const textarea = document.createElement('textarea');
    textarea.style.cssText = `
        flex: 1;
        border: none;
        background: transparent;
        color: #fff;
        padding: 8px;
        resize: none;
        font-family: inherit;
        line-height: 18px;
        white-space: pre-wrap;
        word-wrap: break-word;
    `;

    // Line numbering logic
    function updateLineNumbers() {
        const lines = textarea.value.split('\n');
        let gutterHTML = '';
        let lineNum = 1;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            let number = '';

            // Only number non-empty, non-comment lines
            if (line && !line.startsWith('#')) {
                number = lineNum.toString();
                lineNum++;
            }

            gutterHTML += `<div style="min-height: 18px; padding: 0 2px;">${number || '&nbsp;'}</div>`;
        }

        gutter.innerHTML = gutterHTML;
    }

    // Sync scrolling between gutter and textarea
    textarea.addEventListener('scroll', () => {
        gutter.scrollTop = textarea.scrollTop;
    });

    // Update line numbers on text change
    textarea.addEventListener('input', updateLineNumbers);

    // Handle resize events
    const resizeObserver = new ResizeObserver(updateLineNumbers);
    resizeObserver.observe(textarea);

    container.appendChild(gutter);
    container.appendChild(textarea);

    const widget = node.addDOMWidget(inputName, "code_editor", container, {
        serialize: true,
        getValue: () => textarea.value,
        setValue: (v) => {
            textarea.value = v;
            updateLineNumbers();
        },
        getMinHeight: () => 150,
        getMaxHeight: () => 400
    });

    // Initialize
    textarea.value = "# Default code\nline 1\nline 2";
    updateLineNumbers();

    return { widget };
}
```

## Step 5: Common Patterns and Best Practices

### Layout Patterns
```javascript
// Simple input
const input = document.createElement("input");

// Container with multiple elements
const container = document.createElement("div");
const label = document.createElement("label");
const input = document.createElement("input");
container.appendChild(label);
container.appendChild(input);

// Flexbox layout
container.style.display = "flex";
container.style.flexDirection = "column";
```

### Styling Best Practices
```javascript
// Use CSS strings for complex styling
element.style.cssText = `
    property: value;
    another-property: value;
`;

// ComfyUI-compatible colors
background: #1a1a1a;  // Dark background
background: #2a2a2a;  // Lighter dark
color: #fff;          // White text
border: 1px solid #555; // Subtle border
```

### Event Handling
```javascript
// Input events
element.addEventListener('input', handler);
element.addEventListener('change', handler);
element.addEventListener('focus', handler);
element.addEventListener('blur', handler);

// Custom events
element.addEventListener('mouseenter', handler);
element.addEventListener('mouseleave', handler);
element.addEventListener('click', handler);
```

### Value Management
```javascript
// Always update widget value when DOM changes
input.addEventListener('input', () => {
    widget.value = input.value;
});

// Handle external value changes
setValue(v) {
    input.value = v;
    // Update any dependent UI
    updateRelatedElements();
}
```

## Step 6: Debugging and Testing

### Console Logging
```javascript
console.log("Creating widget for:", inputName);
console.log("Widget created successfully");
console.log("Value changed to:", newValue);
```

### Common Issues

1. **Widget not appearing**: Check type name match between Python and JavaScript
2. **Values not saving**: Ensure `serialize: true` and proper `getValue/setValue`
3. **Layout issues**: Check CSS `box-sizing: border-box` and container dimensions
4. **Scroll sync problems**: Verify event listeners and scroll property access

### Testing Checklist
- [ ] Widget appears when node is created
- [ ] Values are saved and loaded correctly
- [ ] Widget resizes properly with node
- [ ] Event handlers work as expected
- [ ] No console errors
- [ ] Performance is acceptable

## Step 7: File Organization

### Directory Structure
```
your_extension/
├── __init__.py                 # Extension entry point
├── nodes.py                    # Node definitions
└── web/
    └── js/
        └── node.js            # Widget implementations
```

### Extension Registration (`__init__.py`)
```python
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

## Conclusion

Creating custom widgets in ComfyUI involves:

1. **Define custom input type** in Python node
2. **Register widget factory** in JavaScript extension
3. **Create DOM elements** and styling
4. **Handle events** and value synchronization
5. **Return properly wrapped widget**

The key to success is understanding the connection between Python type names and JavaScript widget factories, proper use of `addDOMWidget`, and following ComfyUI's widget lifecycle patterns.

Remember to test thoroughly and refer to ComfyUI's built-in extensions for additional examples and patterns.