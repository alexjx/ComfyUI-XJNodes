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



// Image Navigator Modal Class
class XJImageNavigator {
    constructor(node, directory, subdirectory, images, currentImage) {
        this.node = node;
        this.directory = directory;
        this.subdirectory = subdirectory;
        this.images = images;
        this.currentIndex = Math.max(0, images.indexOf(currentImage));
        this.isVisible = false;

        this.createModal();
        this.bindEvents();
        this.updateDisplay();
    }

    createModal() {
        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'xj-image-navigator-overlay';
        this.overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        // Create main container (fullscreen)
        this.container = document.createElement('div');
        this.container.className = 'xj-image-navigator-container';
        this.container.style.cssText = `
            background: #1a1a1a;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        `;

        // Create image display area (fullscreen, no sidebar)
        this.imageArea = document.createElement('div');
        this.imageArea.className = 'xj-navigator-image-area';
        this.imageArea.style.cssText = `
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        `;

        // Create main image display - maximize viewport
        this.mainImage = document.createElement('img');
        this.mainImage.style.cssText = `
            max-width: 100vw;
            max-height: 100vh;
            width: auto;
            height: auto;
            object-fit: contain;
        `;

        // Create image info overlay
        this.imageInfo = document.createElement('div');
        this.imageInfo.style.cssText = `
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 12px 24px;
            border-radius: 6px;
            font-size: 16px;
            font-family: monospace;
            backdrop-filter: blur(10px);
            text-align: center;
            pointer-events: none;
        `;

        // Keyboard help
        this.keyboardHelp = document.createElement('div');
        this.keyboardHelp.style.cssText = `
            position: absolute;
            top: 30px;
            right: 30px;
            background: rgba(0, 0, 0, 0.8);
            color: #aaa;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-family: monospace;
            backdrop-filter: blur(10px);
            pointer-events: none;
        `;
        this.keyboardHelp.innerHTML = `
            <div style="margin-bottom: 4px;"><strong style="color: #fff;">S</strong> Back | <strong style="color: #fff;">F</strong> Forward</div>
            <div style="margin-bottom: 4px;"><strong style="color: #fff;">Enter / Click</strong> Select & Exit</div>
            <div><strong style="color: #fff;">ESC</strong> Exit</div>
        `;

        // Close button overlay
        this.closeBtn = document.createElement('button');
        this.closeBtn.innerHTML = '✕';
        this.closeBtn.style.cssText = `
            position: absolute;
            top: 30px;
            left: 30px;
            background: rgba(0, 0, 0, 0.8);
            border: none;
            color: #fff;
            font-size: 24px;
            width: 44px;
            height: 44px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(10px);
            transition: background 0.2s;
        `;
        this.closeBtn.onmouseover = () => this.closeBtn.style.background = 'rgba(255, 255, 255, 0.2)';
        this.closeBtn.onmouseout = () => this.closeBtn.style.background = 'rgba(0, 0, 0, 0.8)';

        // Assemble the modal
        this.imageArea.appendChild(this.mainImage);
        this.imageArea.appendChild(this.imageInfo);
        this.imageArea.appendChild(this.keyboardHelp);
        this.imageArea.appendChild(this.closeBtn);

        this.container.appendChild(this.imageArea);
        this.overlay.appendChild(this.container);
    }

    bindEvents() {
        // Keyboard events (only when modal is visible)
        this.keyHandler = (e) => {
            if (!this.isVisible) return;

            const key = e.key.toLowerCase();

            switch(key) {
                case 's':  // Backward
                    e.preventDefault();
                    this.navigate(-1);
                    break;
                case 'f':  // Forward
                    e.preventDefault();
                    this.navigate(1);
                    break;
                case 'enter':
                    e.preventDefault();
                    this.close();
                    break;
                case 'escape':
                    e.preventDefault();
                    this.close();
                    break;
            }
        };

        document.addEventListener('keydown', this.keyHandler);

        // Click anywhere on the dialog to close and select current image
        this.overlay.onclick = (e) => {
            // Close on any click (clicking anywhere selects and exits)
            this.close();
        };

        // Prevent clicks on UI elements from bubbling up and closing
        this.closeBtn.onclick = (e) => {
            e.stopPropagation();
            this.close();
        };
    }

    updateDisplay() {
        const currentImage = this.images[this.currentIndex];

        if (currentImage) {
            // Update main image
            const previewParams = app.getPreviewFormatParam ? app.getPreviewFormatParam() : '';
            this.mainImage.src = `/view?filename=${encodeURIComponent(currentImage)}&type=${this.directory}&subfolder=${encodeURIComponent(this.subdirectory)}${previewParams}&preview=false&channel=rgba&rand=${Math.random()}`;

            // Update image info
            this.imageInfo.textContent = `${this.currentIndex + 1} / ${this.images.length} - ${currentImage}`;
        }
    }

    navigate(direction) {
        const newIndex = this.currentIndex + direction;
        if (newIndex >= 0 && newIndex < this.images.length) {
            this.currentIndex = newIndex;
            this.updateDisplay();
        }
    }

    show() {
        this.isVisible = true;
        document.body.appendChild(this.overlay);

        // Prevent body scroll
        document.body.style.overflow = 'hidden';
    }

    close() {
        // Always update the node with the currently viewed image when exiting
        if (this.images[this.currentIndex]) {
            const imageWidget = this.node.widgets.find(w => w.name === "image");
            if (imageWidget) {
                imageWidget.value = this.images[this.currentIndex];
                imageWidget.callback?.call(this.node, this.images[this.currentIndex]);
            }
        }

        this.isVisible = false;
        if (this.overlay.parentNode) {
            this.overlay.parentNode.removeChild(this.overlay);
        }

        // Restore body scroll
        document.body.style.overflow = '';

        // Clean up event listeners
        document.removeEventListener('keydown', this.keyHandler);
    }
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

                // Capture node reference for use in callbacks
                const node = this;

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
                        // Clean subdirectory: remove leading/trailing slashes
                        let subdirectory = (subdirectoryWidget.value || "").trim();
                        subdirectory = subdirectory.replace(/^\/+|\/+$/g, '');

                        console.debug("updateImagePreview - directory:", directory, "subdirectory:", subdirectory, "filename:", filename);

                        // Create image preview
                        const img = new Image();
                        img.onload = () => {
                            console.debug("Image loaded successfully");
                            app.graph.setDirtyCanvas(true, true);
                        };
                        img.onerror = () => {
                            console.error("Failed to load image from preview URL");
                        };

                        // Build API URL for image with preview optimization
                        // Use preview parameter to get a smaller/compressed version
                        const previewParams = app.getPreviewFormatParam ? app.getPreviewFormatParam() : '';
                        img.src = `/view?filename=${encodeURIComponent(filename)}&type=${directory}&subfolder=${encodeURIComponent(subdirectory)}${previewParams}&preview=true&channel=rgba&rand=${Math.random()}`;

                        console.debug("Preview URL:", img.src);

                        // Set node preview
                        node.imgs = [img];
                        node.imageIndex = 0;
                    };

                    // Function to update image list
                    const updateImageList = async () => {
                        const directory = directoryWidget.value;
                        // Clean subdirectory: remove leading/trailing slashes
                        let subdirectory = (subdirectoryWidget.value || "").trim();
                        subdirectory = subdirectory.replace(/^\/+|\/+$/g, '');
                        const prevValue = imageWidget.value;

                        console.debug("updateImageList - directory:", directory, "subdirectory:", subdirectory);

                        try {
                            const response = await fetch(`/xjnodes/list_images?directory=${encodeURIComponent(directory)}&subdirectory=${encodeURIComponent(subdirectory)}`);
                            if (response.ok) {
                                const data = await response.json();
                                console.debug("API returned:", data);

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
                            } else {
                                console.error("API request failed:", response.status, response.statusText);
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

                    // Add image navigator button
                    const navigatorButton = node.addWidget("button", "🖼️ Browse Images", null, async () => {
                        const directory = directoryWidget.value;
                        let subdirectory = (subdirectoryWidget.value || "").trim();
                        subdirectory = subdirectory.replace(/^\/+|\/+$/g, '');
                        const currentImage = imageWidget.value;

                        try {
                            const response = await fetch(`/xjnodes/list_images?directory=${encodeURIComponent(directory)}&subdirectory=${encodeURIComponent(subdirectory)}`);
                            if (response.ok) {
                                const data = await response.json();
                                if (data.images && data.images.length > 0) {
                                    const navigator = new XJImageNavigator(
                                        node,
                                        directory,
                                        subdirectory,
                                        data.images,
                                        currentImage
                                    );
                                    navigator.show();
                                } else {
                                    alert("No images found in this directory.");
                                }
                            } else {
                                console.error("Failed to load image list for navigator");
                            }
                        } catch (error) {
                            console.error("Error opening image navigator:", error);
                        }
                    });

                    // Style the navigator button
                    navigatorButton.options.y = 8;

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