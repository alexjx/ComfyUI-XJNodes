import { app } from "/scripts/app.js";

// Registry for node update functions (to support refresh)
const registeredNodes = {};

// Function to process text according to XJRandomTextFromList logic
function processTextList(text) {
    const lines = text.split('\n');
    const validLines = [];
    let currentIndex = 1;

    for (const line of lines) {
        const trimmed = line.trim();
        // Skip empty lines and comments (starting with #)
        if (!trimmed || trimmed.startsWith('#')) {
            continue;
        }

        // Remove "- " prefix if present (same as Python: lstrip("- "))
        let cleanLine = trimmed;
        if (cleanLine.startsWith('- ')) {
            cleanLine = cleanLine.substring(2);
        } else if (cleanLine.startsWith('-')) {
            cleanLine = cleanLine.substring(1);
        }

        validLines.push({
            index: currentIndex,
            text: cleanLine,
            originalLine: line
        });
        currentIndex++;
    }

    return validLines;
}



app.registerExtension({
    name: "Comfy.xjnodes",

    getCustomWidgets() {
        return {
            XJ_NUMBERED_LIST(node, inputName) {
                console.log("Creating XJ_NUMBERED_LIST widget for:", inputName);

                // Create container div
                const container = document.createElement("div");
                container.style.cssText = `
                    position: relative;
                    width: 100%;
                    height: 100%;
                    border: 1px solid #555;
                    background: #1a1a1a;
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                    display: flex;
                    overflow: hidden;
                    box-sizing: border-box;
                    cursor: default;
                `;

                // Create line number gutter
                const gutter = document.createElement('div');
                gutter.style.cssText = `
                    width: 50px;
                    height: 100%;
                    background: #2a2a2a;
                    border-right: 1px solid #555;
                    color: #888;
                    text-align: right;
                    padding: 8px 4px;
                    overflow: hidden;
                    user-select: none;
                    line-height: 18px;
                    box-sizing: border-box;
                    cursor: pointer;
                `;

                // Create editable textarea with proper wrapping
                const textarea = document.createElement('textarea');
                textarea.style.cssText = `
                    flex: 1;
                    height: 100%;
                    border: none;
                    outline: none;
                    background: transparent;
                    color: #fff;
                    padding: 8px;
                    resize: none;
                    overflow-y: auto;
                    font-family: inherit;
                    font-size: inherit;
                    line-height: 18px;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    box-sizing: border-box;
                    cursor: text;
                `;
                textarea.wrap = "soft"; // Enable soft wrapping

                // Default value
                const defaultValue = "# This is a comment (will be ignored)\n\n- Option 1: First choice\n- Option 2: Second choice\n- Option 3: Third choice\n\n# Another comment\n- Option 4: Fourth choice";
                textarea.value = defaultValue;

                // Accurate line numbering that measures actual visual wrapping
                function updateLineNumbers() {
                    const text = textarea.value;
                    const lines = text.split('\n');
                    const textareaWidth = textarea.clientWidth;

                    // Preserve current scroll position
                    const currentScrollTop = gutter.scrollTop;

                    // Clear gutter
                    gutter.innerHTML = '';

                    // Get the actual computed styles from textarea
                    const computedStyles = getComputedStyle(textarea);

                    // Create a measurement div with identical styling to textarea
                    const measurer = document.createElement('div');
                    measurer.style.cssText = `
                        position: absolute;
                        top: -9999px;
                        left: -9999px;
                        visibility: hidden;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                        font-family: ${computedStyles.fontFamily};
                        font-size: ${computedStyles.fontSize};
                        font-weight: ${computedStyles.fontWeight};
                        line-height: 18px;
                        width: ${Math.max(textareaWidth - 16, 200)}px;
                        padding: 0;
                        margin: 0;
                        border: none;
                        box-sizing: border-box;
                    `;
                    document.body.appendChild(measurer);

                    let validLineNum = 1;

                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i];
                        const trimmedLine = line.trim();

                        // Determine if this line should have a number
                        let lineNumber = '';
                        if (trimmedLine && !trimmedLine.startsWith('#')) {
                            lineNumber = validLineNum.toString();
                            validLineNum++;
                        }

                        // Measure how many visual lines this logical line actually takes
                        measurer.textContent = line || ' '; // Use space for empty lines
                        const visualHeight = measurer.offsetHeight;
                        const lineHeight = 18;
                        const visualLines = Math.max(1, Math.round(visualHeight / lineHeight));

                        // Create gutter elements for each visual line
                        for (let j = 0; j < visualLines; j++) {
                            const div = document.createElement('div');
                            div.style.cssText = `
                                padding: 0 2px;
                                height: ${lineHeight}px;
                                line-height: ${lineHeight}px;
                                color: ${(j === 0 && lineNumber) ? '#6a9bd7' : 'transparent'};
                            `;
                            // Show line number only on the first visual line of each logical line
                            div.textContent = (j === 0 && lineNumber) ? lineNumber : ' ';

                            // Store the line number directly on the element for easy click handling
                            if (j === 0 && lineNumber) {
                                div.dataset.lineNumber = lineNumber;
                            }

                            gutter.appendChild(div);
                        }
                    }

                    // Restore scroll position
                    gutter.scrollTop = currentScrollTop;

                    // Clean up measurement element
                    document.body.removeChild(measurer);
                }

                // Debounced update - only update after user stops typing
                let updateTimeout;
                textarea.addEventListener('input', () => {
                    // Clear any existing timeout
                    clearTimeout(updateTimeout);

                    // Update choice widget bounds immediately (this is fast)
                    const choiceWidget = node.widgets.find(w => w.name === "choice");
                    if (choiceWidget) {
                        const validLines = processTextList(textarea.value || "");
                        const maxValue = Math.max(1, validLines.length);
                        console.log("Updating choice widget bounds:", maxValue, "valid lines:", validLines.length);
                        choiceWidget.options.max = maxValue;
                        if (choiceWidget.value > maxValue) {
                            choiceWidget.value = maxValue;
                        }
                        if (choiceWidget.value < 1) {
                            choiceWidget.value = 1;
                        }
                        // Force widget to update its UI
                        node.setDirtyCanvas(true);
                    }

                    // Update line numbers after user stops typing for 300ms
                    updateTimeout = setTimeout(() => {
                        updateLineNumbers();
                    }, 300);
                });

                // Handle scroll with proper synchronization
                textarea.addEventListener('scroll', () => {
                    // Sync gutter scroll with textarea scroll
                    gutter.scrollTop = textarea.scrollTop;
                });

                // Update line numbers on resize (affects wrapping)
                const resizeObserver = new ResizeObserver(() => {
                    updateLineNumbers();
                });
                resizeObserver.observe(textarea);

                // Handle gutter clicks to set choice field
                gutter.addEventListener('click', (e) => {
                    const clickedElement = e.target;

                    // Check if the clicked element has a line number stored
                    if (clickedElement.dataset.lineNumber) {
                        const lineNumber = parseInt(clickedElement.dataset.lineNumber, 10);
                        const choiceWidget = node.widgets.find(w => w.name === "choice");
                        if (choiceWidget) {
                            choiceWidget.value = lineNumber;
                        }
                    }
                });

                // Assemble widget
                container.appendChild(gutter);
                container.appendChild(textarea);

                // Create DOM widget using proper ComfyUI method
                const widget = node.addDOMWidget(inputName, "xj_numbered_list", container, {
                    serialize: true,
                    getValue() {
                        return textarea.value;
                    },
                    setValue(v) {
                        textarea.value = v;
                        updateLineNumbers();

                        // Update choice widget bounds when loading from saved workflow
                        const choiceWidget = node.widgets.find(w => w.name === "choice");
                        if (choiceWidget) {
                            const validLines = processTextList(v || "");
                            const maxValue = Math.max(1, validLines.length);
                            console.log("setValue - Updating choice widget bounds:", maxValue, "valid lines:", validLines.length);
                            choiceWidget.options.max = maxValue;
                            if (choiceWidget.value > maxValue) {
                                choiceWidget.value = maxValue;
                            }
                            if (choiceWidget.value < 1) {
                                choiceWidget.value = 1;
                            }
                            // Force widget to update its UI
                            node.setDirtyCanvas(true);
                        }
                    },
                    getMinHeight() {
                        return 150;
                    },
                    getMaxHeight() {
                        return 400;
                    }
                });

                // Set initial value and update
                widget.value = defaultValue;
                updateLineNumbers();

                // Update choice widget bounds with initial content
                const choiceWidget = node.widgets.find(w => w.name === "choice");
                if (choiceWidget) {
                    const validLines = processTextList(defaultValue);
                    const maxValue = Math.max(1, validLines.length);
                    console.log("Initial choice widget bounds:", maxValue, "valid lines:", validLines.length);
                    choiceWidget.options.max = maxValue;
                    if (choiceWidget.value > maxValue) {
                        choiceWidget.value = maxValue;
                    }
                    if (choiceWidget.value < 1) {
                        choiceWidget.value = 1;
                    }
                    // Force widget to update its UI
                    node.setDirtyCanvas(true);
                }

                console.log("XJ_NUMBERED_LIST widget created successfully");

                return { widget };
            }
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "XJRandomTextFromList") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = originalNodeCreated?.apply(this, arguments);

                // Fix choice widget initialization
                const choiceWidget = this.widgets.find(w => w.name === "choice");
                if (choiceWidget) {
                    choiceWidget.options.min = 1;
                    if (choiceWidget.value < 1) {
                        choiceWidget.value = 1;
                    }
                    // Set initial max to a reasonable value, will be updated when text changes
                    choiceWidget.options.max = 100;
                }

                return result;
            };
        }

        // Handle XJRandomTextFromFile node
        if (nodeData.name === "XJRandomTextFromFile") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = originalNodeCreated?.apply(this, arguments);

                const filePathWidget = this.widgets.find(w => w.name === "file_path");
                const choiceWidget = this.widgets.find(w => w.name === "choice");

                if (filePathWidget && choiceWidget) {
                    // Function to update choice max value based on selected file
                    const updateChoiceMax = async (filename) => {
                        if (!filename) {
                            choiceWidget.options.max = 1;
                            return;
                        }

                        try {
                            const response = await fetch(`/xjnodes/get_text_file_lines?filename=${encodeURIComponent(filename)}`);
                            if (response.ok) {
                                const data = await response.json();
                                choiceWidget.options.max = data.max_choice;

                                // Clamp current choice value to valid range
                                if (choiceWidget.value > data.max_choice) {
                                    choiceWidget.value = data.max_choice;
                                }
                                if (choiceWidget.value < 1) {
                                    choiceWidget.value = 1;
                                }

                                // Force node to redraw
                                this.setDirtyCanvas(true, true);
                            }
                        } catch (error) {
                            console.error("Error fetching text file line count:", error);
                        }
                    };

                    // Store original callback
                    const originalCallback = filePathWidget.callback;

                    // Override the callback to update choice max when file changes
                    filePathWidget.callback = function(value) {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        updateChoiceMax.call(nodeType.prototype.onNodeCreated, value);
                    };

                    // Initialize choice widget settings
                    choiceWidget.options.min = 1;
                    if (choiceWidget.value < 1) {
                        choiceWidget.value = 1;
                    }

                    // Update max value on initial load
                    if (filePathWidget.value) {
                        updateChoiceMax.call(this, filePathWidget.value);
                    }
                }

                return result;
            };
        }

        // Handle XJLoadImageWithMetadata node
        if (nodeData.name === "XJLoadImageWithMetadata") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = async function () {
                const result = originalNodeCreated?.apply(this, arguments);

                const directoryWidget = this.widgets.find(w => w.name === "directory");
                const subdirectoryWidget = this.widgets.find(w => w.name === "subdirectory");
                const imageWidget = this.widgets.find(w => w.name === "image");

                if (directoryWidget && subdirectoryWidget && imageWidget) {
                    // Ensure the image widget has proper image_upload flag for preview
                    imageWidget.options.image_upload = true;

                    // Function to load and display image preview
                    const updateImagePreview = (filename) => {
                        if (!filename) {
                            node.imgs = null;
                            return;
                        }

                        const directory = directoryWidget.value;
                        const subdirectory = subdirectoryWidget.value || "";

                        // Create image preview
                        const img = new Image();
                        img.onload = () => {
                            app.graph.setDirtyCanvas(true, true);
                        };

                        // Build API URL for image with preview optimization
                        // Use preview parameter to get a smaller/compressed version
                        const previewParams = app.getPreviewFormatParam ? app.getPreviewFormatParam() : '';
                        img.src = `/view?filename=${encodeURIComponent(filename)}&type=${directory}&subfolder=${encodeURIComponent(subdirectory)}${previewParams}&preview=true&channel=rgba&rand=${Math.random()}`;

                        // Set node preview
                        node.imgs = [img];
                        node.imageIndex = 0;
                    };

                    // Function to update image list
                    const updateImageList = async () => {
                        const directory = directoryWidget.value;
                        const subdirectory = subdirectoryWidget.value || "";
                        const prevValue = imageWidget.value;

                        try {
                            const response = await fetch(`/xjnodes/list_images?directory=${encodeURIComponent(directory)}&subdirectory=${encodeURIComponent(subdirectory)}`);
                            if (response.ok) {
                                const data = await response.json();

                                // Update image widget options (just use the filenames directly)
                                imageWidget.options.values = data.images.length > 0 ? data.images : [""];

                                // Preserve previous value if it still exists in the list
                                if (data.images.length > 0 && data.images.includes(prevValue)) {
                                    imageWidget.value = prevValue;
                                } else if (data.images.length > 0) {
                                    imageWidget.value = data.images[0];
                                } else {
                                    imageWidget.value = "";
                                }

                                // Update image preview
                                updateImagePreview(imageWidget.value);

                                console.debug("Updated imageWidget.options.values:", imageWidget.options.values);
                                console.debug("Updated imageWidget.value:", imageWidget.value);
                            }
                        } catch (error) {
                            console.error("Error fetching image list:", error);
                        }
                    };

                    // Hook into directory widget change
                    directoryWidget.callback = updateImageList;

                    // Hook into subdirectory widget change
                    subdirectoryWidget.callback = updateImageList;

                    // Hook into image widget change to update preview
                    const originalImageCallback = imageWidget.callback;
                    imageWidget.callback = function(value) {
                        if (originalImageCallback) {
                            originalImageCallback.apply(this, arguments);
                        }
                        updateImagePreview(value);
                    };

                    // Dummy function to get actual values from web page
                    const dummy = async () => {
                        // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
                    };

                    // Initial update
                    await dummy(); // this will cause the widgets to obtain the actual value from web page.
                    await updateImageList();

                    // Save the updateImageList method for refreshing later
                    if (!registeredNodes[nodeData.name]) {
                        registeredNodes[nodeData.name] = [];
                    }
                    registeredNodes[nodeData.name].push(updateImageList);
                }

                return result;
            };

            // Refresh existing nodes when node is re-registered (on refresh)
            if (registeredNodes[nodeData.name]) {
                for (const updateImageList of registeredNodes[nodeData.name]) {
                    await updateImageList();
                }
            }
        }
    }
});