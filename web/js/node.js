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

        // Create main container (fullscreen with flex layout)
        this.container = document.createElement('div');
        this.container.className = 'xj-image-navigator-container';
        this.container.style.cssText = `
            background: #1a1a1a;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: row;
            overflow: hidden;
        `;

        // Create image display area (left side - 80%)
        this.imageArea = document.createElement('div');
        this.imageArea.className = 'xj-navigator-image-area';
        this.imageArea.style.cssText = `
            width: 80%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        `;

        // Create main image display
        this.mainImage = document.createElement('img');
        this.mainImage.style.cssText = `
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            object-fit: contain;
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
            opacity: 0.5;
        `;
        this.keyboardHelp.innerHTML = `
            <div style="margin-bottom: 4px;"><strong style="color: #fff;">A</strong> Back | <strong style="color: #fff;">D</strong> Forward</div>
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

        // Create metadata panel (right side - 20%)
        this.metadataPanel = document.createElement('div');
        this.metadataPanel.className = 'xj-navigator-metadata-panel';
        this.metadataPanel.style.cssText = `
            width: 20%;
            height: 100vh;
            background: #1e1e1e;
            border-left: 1px solid #333;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 0;
            color: #ddd;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 13px;
        `;

        // Metadata panel header (sticky) - includes image info
        this.metadataHeader = document.createElement('div');
        this.metadataHeader.style.cssText = `
            padding: 20px;
            position: sticky;
            top: 0;
            background: #1e1e1e;
            border-bottom: 1px solid #333;
            z-index: 1;
        `;
        this.metadataHeader.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; color: #fff; font-size: 16px;">📊 Generation Info</h3>
                <button id="delete-image-btn" style="
                    background: #d32f2f;
                    border: none;
                    color: #fff;
                    padding: 6px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: background 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 4px;
                " title="Delete this image">
                    🗑️ Delete
                </button>
            </div>
            <div id="image-info-text" style="
                font-size: 12px;
                color: #888;
                font-family: monospace;
                word-break: break-word;
            ">Loading...</div>
        `;

        // Get reference to image info text element
        this.imageInfoText = null; // Will be set after appending

        // Metadata content area
        this.metadataContent = document.createElement('div');
        this.metadataContent.id = 'metadata-content';
        this.metadataContent.style.cssText = `
            padding: 20px;
        `;
        this.metadataContent.innerHTML = `
            <div style="text-align: center; padding: 40px 0; color: #666;">
                Loading metadata...
            </div>
        `;

        this.metadataPanel.appendChild(this.metadataHeader);
        this.metadataPanel.appendChild(this.metadataContent);

        // Get reference to image info text and delete button after DOM is ready
        this.imageInfoText = this.metadataHeader.querySelector('#image-info-text');
        this.deleteButton = this.metadataHeader.querySelector('#delete-image-btn');

        // Assemble the modal
        this.imageArea.appendChild(this.mainImage);
        this.imageArea.appendChild(this.keyboardHelp);
        this.imageArea.appendChild(this.closeBtn);

        this.container.appendChild(this.imageArea);
        this.container.appendChild(this.metadataPanel);
        this.overlay.appendChild(this.container);
    }

    bindEvents() {
        // Keyboard events (only when modal is visible)
        this.keyHandler = (e) => {
            if (!this.isVisible) return;

            const key = e.key.toLowerCase();

            switch(key) {
                case 'a':  // Backward
                    e.preventDefault();
                    this.navigate(-1);
                    break;
                case 'd':  // Forward
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

        // Click on image area to close and select current image
        this.imageArea.onclick = (e) => {
            // Only close if clicking directly on the image area (not on overlays)
            if (e.target === this.imageArea || e.target === this.mainImage) {
                this.close();
            }
        };

        // Prevent clicks on metadata panel from closing
        this.metadataPanel.onclick = (e) => {
            e.stopPropagation();
        };

        // Close button
        this.closeBtn.onclick = (e) => {
            e.stopPropagation();
            this.close();
        };

        // Delete button
        this.deleteButton.onclick = async (e) => {
            e.stopPropagation();
            await this.handleDelete();
        };

        // Hover effect for delete button
        this.deleteButton.onmouseover = () => {
            this.deleteButton.style.background = '#b71c1c';
        };
        this.deleteButton.onmouseout = () => {
            this.deleteButton.style.background = '#d32f2f';
        };
    }

    updateDisplay() {
        const currentImage = this.images[this.currentIndex];

        if (currentImage) {
            // Update main image
            const previewParams = app.getPreviewFormatParam ? app.getPreviewFormatParam() : '';
            this.mainImage.src = `/view?filename=${encodeURIComponent(currentImage)}&type=${this.directory}&subfolder=${encodeURIComponent(this.subdirectory)}${previewParams}&preview=false&channel=rgba&rand=${Math.random()}`;

            // Update image info in metadata panel header
            if (this.imageInfoText) {
                this.imageInfoText.textContent = `${this.currentIndex + 1} / ${this.images.length} - ${currentImage}`;
            }

            // Load metadata for current image
            this.loadMetadata(currentImage);
        }
    }

    async loadMetadata(filename) {
        this.metadataContent.innerHTML = `
            <div style="text-align: center; padding: 40px 0; color: #666;">
                Loading metadata...
            </div>
        `;

        try {
            const response = await fetch(
                `/xjnodes/get_image_metadata?` +
                `directory=${encodeURIComponent(this.directory)}&` +
                `subdirectory=${encodeURIComponent(this.subdirectory)}&` +
                `filename=${encodeURIComponent(filename)}`
            );

            if (response.ok) {
                const data = await response.json();
                this.renderMetadata(data);
            } else {
                this.renderError("Failed to load metadata");
            }
        } catch (error) {
            console.error("Failed to load metadata:", error);
            this.renderError("Error loading metadata");
        }
    }

    renderMetadata(data) {
        if (!data.success || !data.parsed || !data.parsed.nodes || data.parsed.nodes.length === 0) {
            this.metadataContent.innerHTML = `
                <div style="text-align: center; padding: 40px 20px; color: #666;">
                    No generation parameters found
                </div>
            `;
            return;
        }

        const nodes = data.parsed.nodes;
        let html = "";

        // Render each node as a section
        nodes.forEach((node) => {
            html += this.renderNode(node);
        });

        this.metadataContent.innerHTML = html;
    }

    renderNode(node) {
        const title = node.title !== node.type ?
            `${node.icon} ${node.type} #${node.id} (${node.title})` :
            `${node.icon} ${node.type} #${node.id}`;

        return `
            <div style="margin-bottom: 24px;">
                <div style="
                    color: #fff;
                    font-weight: 600;
                    margin-bottom: 12px;
                    font-size: 14px;
                    padding-bottom: 8px;
                    border-bottom: 1px solid #333;
                ">${this.escapeHtml(title)}</div>

                <div style="margin-left: 8px;">
                    ${this.renderParams(node.params)}
                </div>
            </div>
        `;
    }

    renderParams(params) {
        let html = "";

        // Define display order and labels (NO seed/vae)
        const paramOrder = [
            { key: "checkpoint", label: "Checkpoint" },
            { key: "unet", label: "Diffusion Model" },
            { key: "lora", label: "LoRA" },
            { key: "lora_strength", label: "Strength" },
            { key: "sampler", label: "Sampler" },
            { key: "scheduler", label: "Scheduler" },
            { key: "steps", label: "Steps" },
            { key: "cfg", label: "CFG" },
            { key: "denoise", label: "Denoise" },
            { key: "positive", label: "Positive" },
            { key: "negative", label: "Negative" },
        ];

        paramOrder.forEach(({ key, label }) => {
            if (params.hasOwnProperty(key)) {
                const value = params[key];

                // Special rendering for prompts (multi-line)
                if (key === "positive" || key === "negative") {
                    html += this.renderPromptParam(label, value);
                } else {
                    html += this.renderParam(label, value);
                }
            }
        });

        return html;
    }

    renderParam(label, value) {
        return `
            <div style="
                display: flex;
                margin-bottom: 8px;
                line-height: 1.5;
            ">
                <span style="
                    color: #888;
                    min-width: 90px;
                    flex-shrink: 0;
                ">${label}:</span>
                <span style="
                    color: #ddd;
                    word-break: break-word;
                ">${this.escapeHtml(String(value))}</span>
            </div>
        `;
    }

    renderPromptParam(label, text) {
        // Truncate very long prompts
        const maxLength = 200;
        const displayText = text.length > maxLength ?
            text.substring(0, maxLength) + "..." : text;

        return `
            <div style="margin-bottom: 12px;">
                <div style="color: #888; margin-bottom: 4px;">${label}:</div>
                <div style="
                    color: #ddd;
                    background: #2a2a2a;
                    padding: 10px;
                    border-radius: 4px;
                    border-left: 3px solid #444;
                    white-space: pre-wrap;
                    word-break: break-word;
                    font-size: 12px;
                    line-height: 1.6;
                ">${this.escapeHtml(displayText)}</div>
            </div>
        `;
    }

    renderError(message) {
        this.metadataContent.innerHTML = `
            <div style="text-align: center; padding: 40px 20px; color: #666;">
                ${this.escapeHtml(message)}
            </div>
        `;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    navigate(direction) {
        const newIndex = this.currentIndex + direction;
        if (newIndex >= 0 && newIndex < this.images.length) {
            this.currentIndex = newIndex;
            this.updateDisplay();
        }
    }

    async handleDelete() {
        const currentImage = this.images[this.currentIndex];

        if (!currentImage) {
            return;
        }

        // Show confirmation dialog
        const confirmed = confirm(`Are you sure you want to delete "${currentImage}"?\n\nThis action cannot be undone.`);

        if (!confirmed) {
            return;
        }

        try {
            // Call delete API
            const response = await fetch('/xjnodes/delete_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    directory: this.directory,
                    subdirectory: this.subdirectory,
                    filename: currentImage
                })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                // Remove the image from our list
                this.images.splice(this.currentIndex, 1);

                // Update the node's image widget to remove the deleted image
                const imageWidget = this.node.widgets.find(w => w.name === "image");
                if (imageWidget && imageWidget.options && imageWidget.options.values) {
                    const index = imageWidget.options.values.indexOf(currentImage);
                    if (index > -1) {
                        imageWidget.options.values.splice(index, 1);
                    }
                }

                // If no more images, close the modal
                if (this.images.length === 0) {
                    alert('No more images in this directory.');
                    this.close();
                    return;
                }

                // Adjust currentIndex if we deleted the last image
                if (this.currentIndex >= this.images.length) {
                    this.currentIndex = this.images.length - 1;
                }

                // Update display to show next/previous image
                this.updateDisplay();

                // Update the node's image widget value
                if (imageWidget && this.images[this.currentIndex]) {
                    imageWidget.value = this.images[this.currentIndex];
                }
            } else {
                alert(`Failed to delete image: ${result.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error deleting image:', error);
            alert(`Error deleting image: ${error.message}`);
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

// Image Pair Compare Widget
class XJImagePairCompareWidget {
    constructor(name, node) {
        this.name = name;
        this.type = "custom";
        this.node = node;
        this.selectedPairIndex = 0;
        this.imagesA = [];
        this.imagesB = [];
        this.pairCount = 0;
        this.hitAreas = {};
        this._value = { pairs: [] };
        this.pointerOverPos = [0, 0];
        this.isPointerOver = false;
        this.isPointerDown = false;
        // Store the actual Y position where we start drawing (set by ComfyUI)
        this.startY = 0;
    }

    get value() {
        return this._value;
    }

    set value(v) {
        this._value = v || { pairs: [] };
        this.loadImages();
    }

    setSelected(index) {
        if (index < 0 || index >= this.pairCount) return;
        this.selectedPairIndex = index;
        this.loadSelectedImages();
    }

    loadImages() {
        this.imagesA = [];
        this.imagesB = [];
        this.pairCount = Math.min(this._value.pairs.length, 100); // Sanity limit

        for (let i = 0; i < this.pairCount; i++) {
            const pair = this._value.pairs[i];
            if (pair.a_url) {
                const imgA = new Image();
                imgA.onload = () => {
                    if (this.node) {
                        this.node.setDirtyCanvas(true, true);
                    }
                };
                imgA.src = pair.a_url;
                this.imagesA.push(imgA);
            }
            if (pair.b_url) {
                const imgB = new Image();
                imgB.onload = () => {
                    if (this.node) {
                        this.node.setDirtyCanvas(true, true);
                    }
                };
                imgB.src = pair.b_url;
                this.imagesB.push(imgB);
            }
        }
    }

    loadSelectedImages() {
        const index = this.selectedPairIndex;
        if (index < this.imagesA.length && !this.imagesA[index].complete) {
            // Trigger reload if needed
        }
        if (index < this.imagesB.length && !this.imagesB[index].complete) {
            // Trigger reload if needed
        }
    }

    draw(ctx, node, width, y) {
        this.hitAreas = {};
        this.startY = y; // Store the starting Y position for click detection

        // Draw numeric tabs if more than 1 pair
        if (this.pairCount > 1) {
            y = this.drawTabs(ctx, node, width, y);
        }

        // Draw images and size label
        y = this.drawImages(ctx, node, width, y);

        return y;
    }

    drawTabs(ctx, node, width, y) {
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "14px Arial";

        const tabHeight = 20;
        const tabPadding = 8;  // Padding on each side of text
        const spacing = 6;
        const tabData = [];

        // Measure all tabs first
        let totalWidth = 0;
        for (let i = 0; i < this.pairCount; i++) {
            const text = (i + 1).toString();
            const textWidth = ctx.measureText(text).width;
            const tabWidth = textWidth + tabPadding * 2;
            tabData.push({ index: i, text, width: tabWidth });
            totalWidth += tabWidth + spacing;
        }
        totalWidth -= spacing; // Remove last spacing

        // Center tabs - calculate starting x
        let x = Math.max(5, (width - totalWidth) / 2);

        // Draw tabs
        for (const tab of tabData) {
            const isSelected = tab.index === this.selectedPairIndex;

            // Draw tab border only (no fill)
            const radius = 4;
            ctx.beginPath();
            ctx.roundRect(x, y, tab.width, tabHeight, radius);

            // Border color
            if (isSelected) {
                ctx.strokeStyle = "rgba(100, 150, 255, 1)";
            } else {
                ctx.strokeStyle = "rgba(150, 150, 150, 0.6)";
            }
            ctx.lineWidth = isSelected ? 2 : 1;
            ctx.stroke();

            // Draw text (offset down by 2px for better visual centering)
            ctx.fillStyle = isSelected ? "rgba(100, 150, 255, 1)" : "rgba(180, 180, 180, 0.8)";
            ctx.fillText(tab.text, x + tab.width / 2, y + tabHeight / 2 + 1);

            // Store hit area for click detection
            this.hitAreas[`tab_${tab.index}`] = {
                bounds: [x, y, tab.width, tabHeight],
                data: tab.index,
            };

            x += tab.width + spacing;
        }

        return y + tabHeight + 6;
    }

    drawImages(ctx, node, width, y) {
        if (this.pairCount === 0) {
            return y;
        }

        const imgA = this.imagesA[this.selectedPairIndex];
        const imgB = this.imagesB[this.selectedPairIndex];

        if (!imgA || !imgB || !imgA.complete || !imgB.complete) {
            return y;
        }

        const imageA = imgA;
        const imageB = imgB;

        if (!imageA.naturalWidth || !imageA.naturalHeight) {
            return y;
        }

        const mode = node.properties["comparer_mode"] || "Slide";
        const [nodeWidth, nodeHeight] = node.size;
        const availableHeight = nodeHeight - y - 35; // Reserve space for size label

        // Calculate dimensions to fit (maintain aspect ratio)
        const imageAspect = imageA.naturalWidth / imageA.naturalHeight;
        const widgetAspect = nodeWidth / availableHeight;

        let targetWidth, targetHeight;

        if (imageAspect > widgetAspect) {
            // Image is wider - fit to width
            targetWidth = nodeWidth;
            targetHeight = nodeWidth / imageAspect;
        } else {
            // Image is taller - fit to height
            targetHeight = availableHeight;
            targetWidth = availableHeight * imageAspect;
        }

        // Center the image
        const destX = (nodeWidth - targetWidth) / 2;
        const destY = y + (availableHeight - targetHeight) / 2;

        // Draw image A (base image)
        ctx.save();
        ctx.drawImage(imageA, destX, destY, targetWidth, targetHeight);
        ctx.restore();

        // Draw image B (overlay)
        if (mode === "Click") {
            if (this.isPointerDown) {
                ctx.save();
                ctx.drawImage(imageB, destX, destY, targetWidth, targetHeight);
                ctx.restore();
            }
        } else if (mode === "Slide" && this.isPointerOver) {
            // Calculate crop position based on mouse
            const cropX = this.pointerOverPos[0];

            // Calculate source and destination dimensions for image B
            // Only draw image B from cropX to the right edge
            if (cropX > destX && cropX < destX + targetWidth) {
                const sourceX = (cropX - destX) * (imageB.naturalWidth / targetWidth);
                const sourceWidth = imageB.naturalWidth - sourceX;
                const destWidth = targetWidth - (cropX - destX);

                if (destWidth > 0 && sourceWidth > 0) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(cropX, destY, destWidth, targetHeight);
                    ctx.clip();
                    ctx.drawImage(imageB, sourceX, 0, sourceWidth, imageB.naturalHeight, cropX, destY, destWidth, targetHeight);
                    ctx.restore();

                    // Draw divider line
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(cropX, destY);
                    ctx.lineTo(cropX, destY + targetHeight);
                    ctx.globalCompositeOperation = "difference";
                    ctx.strokeStyle = "rgba(255, 255, 255, 1)";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.restore();
                }
            }
        }

        // Draw size label (same style as tabs)
        ctx.save();
        ctx.fillStyle = "rgba(180, 180, 180, 1)";
        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";

        // Show different format based on whether images have same size
        let sizeText;
        if (imageA.naturalWidth === imageB.naturalWidth && imageA.naturalHeight === imageB.naturalHeight) {
            // Same size - show single format
            sizeText = `${imageA.naturalWidth} x ${imageA.naturalHeight}`;
        } else {
            // Different sizes - show both
            sizeText = `${imageA.naturalWidth} x ${imageA.naturalHeight} | ${imageB.naturalWidth} x ${imageB.naturalHeight}`;
        }

        ctx.fillText(sizeText, nodeWidth / 2, destY + targetHeight + 8);
        ctx.restore();

        return y;
    }

    onMouseMove(event, pos, node) {
        this.pointerOverPos = [...pos];
    }

    onMouseEnter() {
        this.isPointerOver = true;
    }

    onMouseLeave() {
        this.isPointerOver = false;
    }

    setIsPointerDown(down) {
        this.isPointerDown = down && app.canvas.pointer_is_down;
    }

    computeSize(width) {
        // Base height for tabs + images + size label
        const tabHeight = this.pairCount > 1 ? 20 : 0;
        return [width, tabHeight + 250];
    }

    serializeValue(node, index) {
        const v = [];
        for (const pair of this._value.pairs) {
            v.push({
                a_url: pair.a_url,
                b_url: pair.b_url,
            });
        }
        return { pairs: v, selectedIndex: this.selectedPairIndex };
    }

    // Handle mouse events using widget's mouse() method (like rgthree)
    // This receives pos in the correct coordinate space matching the draw() method
    mouse(event, pos, node) {
        if (event.type === "pointerdown") {
            // Check if click is within any tab hit area
            for (const [key, area] of Object.entries(this.hitAreas)) {
                if (key.startsWith("tab_")) {
                    const [bx, by, bw, bh] = area.bounds;

                    // Check if click is within this tab's bounds
                    if (pos[0] >= bx && pos[0] <= bx + bw &&
                        pos[1] >= by && pos[1] <= by + bh) {
                        this.setSelected(area.data);
                        node.setDirtyCanvas(true, true);
                        return true; // Event handled
                    }
                }
            }
        }

        // Let other mouse handling continue
        return false;
    }
}

app.registerExtension({
    name: "Comfy.xjnodes",

    getCustomWidgets() {
        return {
            XJ_NUMBERED_LIST(node, inputName) {
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
            const originalOnConfigure = nodeType.prototype.onConfigure;

            nodeType.prototype.onNodeCreated = async function () {
                const result = originalNodeCreated?.apply(this, arguments);

                // Capture node reference for use in callbacks
                const node = this;

                const directoryWidget = this.widgets.find(w => w.name === "directory");
                const subdirectoryWidget = this.widgets.find(w => w.name === "subdirectory");
                const imageWidget = this.widgets.find(w => w.name === "image");

                if (directoryWidget && subdirectoryWidget && imageWidget) {
                    // CRITICAL FIX: If we have a saved image value in widgets_values, add it to the combo options
                    // This must happen BEFORE dummy() is called, otherwise the value gets rejected
                    if (this.widgets_values && this.widgets_values[2]) {
                        const savedImageValue = this.widgets_values[2];

                        // Add the saved value to the combo options so it won't be rejected
                        if (!imageWidget.options.values.includes(savedImageValue)) {
                            imageWidget.options.values.push(savedImageValue);
                        }

                        // Set the value directly
                        imageWidget.value = savedImageValue;
                    }

                    // Ensure the image widget properly serializes its value to the workflow
                    // This is critical for the widget value to persist across workflow saves/loads
                    imageWidget.serializeValue = function() {
                        return this.value;
                    };

                    // Subdirectory dropdown helper - works like Windows file dialog
                    let subdirDropdown = null;

                    // Function to close dropdown
                    const closeDropdown = () => {
                        if (subdirDropdown && subdirDropdown.parentNode) {
                            subdirDropdown.parentNode.removeChild(subdirDropdown);
                            subdirDropdown = null;
                        }
                    };

                    // Function to create and show dropdown with subdirectories
                    const showSubdirectoryDropdown = async (currentPath, clickEvent) => {
                        const directory = directoryWidget.value;

                        // Clean current path: remove leading/trailing slashes
                        currentPath = (currentPath || "").trim().replace(/^\/+|\/+$/g, '');

                        try {
                            const response = await fetch(`/xjnodes/list_subdirectories?directory=${encodeURIComponent(directory)}&current_path=${encodeURIComponent(currentPath)}`);

                            if (!response.ok) {
                                console.error("Failed to fetch subdirectories, status:", response.status);
                                return;
                            }

                            const data = await response.json();
                            const subdirs = data.subdirectories || [];

                            // Close existing dropdown
                            closeDropdown();

                            // Only skip showing dropdown if no subdirs AND at root (can't go up)
                            if (subdirs.length === 0 && !currentPath) {
                                return;
                            }

                            // Create dropdown
                            subdirDropdown = document.createElement('div');
                            subdirDropdown.className = 'xj-subdirectory-dropdown';
                            subdirDropdown.style.cssText = `
                                position: fixed;
                                background: #2a2a2a;
                                border: 1px solid #555;
                                border-radius: 4px;
                                max-height: 300px;
                                overflow-y: auto;
                                z-index: 99999;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                                min-width: 300px;
                                left: 100px;
                                top: 100px;
                            `;

                            // Add ".." parent directory option if not at root
                            if (currentPath) {
                                const parentOption = document.createElement('div');
                                parentOption.textContent = '.. (Parent Directory)';
                                parentOption.style.cssText = `
                                    padding: 8px 12px;
                                    cursor: pointer;
                                    color: #ffa500;
                                    border-bottom: 1px solid #555;
                                    font-weight: bold;
                                `;

                                parentOption.onmouseover = () => {
                                    parentOption.style.background = '#3a3a3a';
                                };
                                parentOption.onmouseout = () => {
                                    parentOption.style.background = 'transparent';
                                };

                                parentOption.onclick = async (e) => {
                                    e.stopPropagation();

                                    // Go to parent directory
                                    const pathParts = currentPath.split('/').filter(p => p);
                                    pathParts.pop(); // Remove last part
                                    const parentPath = pathParts.join('/');

                                    // Update widget value
                                    subdirectoryWidget.value = parentPath;

                                    // Update image list
                                    updateImageList();

                                    // Force canvas redraw
                                    app.graph.setDirtyCanvas(true, true);

                                    // Refresh dropdown with new location (keep it open)
                                    await showSubdirectoryDropdown(parentPath, clickEvent);
                                };

                                subdirDropdown.appendChild(parentOption);
                            }

                            // Add subdirectory options
                            subdirs.forEach((subdir) => {
                                const option = document.createElement('div');
                                option.textContent = subdir;
                                option.style.cssText = `
                                    padding: 8px 12px;
                                    cursor: pointer;
                                    color: #fff;
                                    border-bottom: 1px solid #333;
                                `;

                                option.onmouseover = () => {
                                    option.style.background = '#3a3a3a';
                                };
                                option.onmouseout = () => {
                                    option.style.background = 'transparent';
                                };

                                option.onclick = async (e) => {
                                    e.stopPropagation();

                                    // Build new path
                                    let newPath;
                                    if (currentPath) {
                                        newPath = currentPath + '/' + subdir;
                                    } else {
                                        newPath = subdir;
                                    }

                                    // Update widget value
                                    subdirectoryWidget.value = newPath;

                                    // Update image list (trigger the original subdirectory callback)
                                    updateImageList();

                                    // Force canvas redraw
                                    app.graph.setDirtyCanvas(true, true);

                                    // Refresh dropdown with new location (keep it open)
                                    await showSubdirectoryDropdown(newPath, clickEvent);
                                };

                                subdirDropdown.appendChild(option);
                            });

                            // Append to DOM
                            document.body.appendChild(subdirDropdown);

                            // Position dropdown below/right of the button click
                            let x, y;

                            if (clickEvent && clickEvent.clientX && clickEvent.clientY) {
                                // Position relative to click location (bottom-right)
                                x = clickEvent.clientX;
                                y = clickEvent.clientY + 10; // 10px below click
                            } else {
                                // Fallback: center-left of screen
                                const viewportWidth = window.innerWidth;
                                const viewportHeight = window.innerHeight;
                                x = Math.max(20, (viewportWidth - 300) / 4);
                                y = Math.max(100, viewportHeight / 4);
                            }

                            // Make sure dropdown stays within viewport
                            const dropdownWidth = 300;
                            const dropdownHeight = 300;
                            const viewportWidth = window.innerWidth;
                            const viewportHeight = window.innerHeight;

                            // Adjust if too far right
                            if (x + dropdownWidth > viewportWidth - 20) {
                                x = viewportWidth - dropdownWidth - 20;
                            }

                            // Adjust if too far down
                            if (y + dropdownHeight > viewportHeight - 20) {
                                y = Math.max(20, viewportHeight - dropdownHeight - 20);
                            }

                            subdirDropdown.style.left = x + 'px';
                            subdirDropdown.style.top = y + 'px';

                        } catch (error) {
                            console.error("Error fetching subdirectories:", error);
                        }
                    };

                    // Close dropdown when clicking outside
                    const globalClickHandler = (e) => {
                        if (subdirDropdown && !subdirDropdown.contains(e.target)) {
                            closeDropdown();
                        }
                    };
                    document.addEventListener('click', globalClickHandler);

                    // Store preview image in custom property (not node.imgs to avoid ImagePreviewWidget)
                    node.previewImage = null;

                    // Custom draw function for image preview
                    const originalOnDrawForeground = node.onDrawForeground;
                    node.onDrawForeground = function(ctx) {
                        // Call original draw function if it exists
                        if (originalOnDrawForeground) {
                            originalOnDrawForeground.apply(this, arguments);
                        }

                        // Draw our custom image preview
                        if (node.previewImage && node.previewImage.complete && node.previewImage.naturalWidth > 0) {
                            const img = node.previewImage;

                            // Calculate available space (below widgets)
                            // Find the bottom-most widget position
                            let widgetsEndY = LiteGraph.NODE_TITLE_HEIGHT || 30;

                            if (node.widgets && node.widgets.length > 0) {
                                // Use the last_y property from the last widget if available
                                const lastWidget = node.widgets[node.widgets.length - 1];
                                if (lastWidget.last_y !== undefined) {
                                    widgetsEndY = lastWidget.last_y;
                                } else {
                                    // Fallback: sum all widget heights
                                    widgetsEndY = LiteGraph.NODE_TITLE_HEIGHT || 30;
                                    for (const w of node.widgets) {
                                        const height = w.computeSize ? w.computeSize()[1] : (LiteGraph.NODE_WIDGET_HEIGHT || 25);
                                        widgetsEndY += height;
                                    }
                                }
                            }

                            // Add generous margins to separate from widgets
                            const topMargin = 35;
                            const bottomMargin = 20;
                            const sideMargin = 10;

                            const availableY = widgetsEndY + topMargin;
                            const availableHeight = node.size[1] - availableY - bottomMargin;
                            const availableWidth = node.size[0] - sideMargin * 2;

                            if (availableHeight > 20 && availableWidth > 20) {
                                // Calculate scaled dimensions to fit
                                const imgAspect = img.naturalWidth / img.naturalHeight;
                                let drawWidth = availableWidth;
                                let drawHeight = drawWidth / imgAspect;

                                if (drawHeight > availableHeight) {
                                    drawHeight = availableHeight;
                                    drawWidth = drawHeight * imgAspect;
                                }

                                // Center the image horizontally, position vertically in available space
                                const x = sideMargin + (availableWidth - drawWidth) / 2;
                                const y = availableY + (availableHeight - drawHeight) / 2;

                                // Draw image
                                ctx.save();
                                ctx.drawImage(img, x, y, drawWidth, drawHeight);
                                ctx.restore();

                                // Draw image dimensions text
                                ctx.save();
                                ctx.fillStyle = "#AAA";
                                ctx.font = "10px monospace";
                                ctx.textAlign = "center";
                                const dimText = `${img.naturalWidth} × ${img.naturalHeight}`;
                                ctx.fillText(dimText, node.size[0] / 2, y + drawHeight + 12);
                                ctx.restore();
                            }
                        }
                    };

                    // Function to load and display image preview
                    const updateImagePreview = (filename) => {
                        // Clear preview if no valid filename
                        if (!filename || filename === "" || filename === undefined || filename === null) {
                            node.previewImage = null;
                            app.graph.setDirtyCanvas(true, true);
                            return;
                        }

                        const directory = directoryWidget.value;
                        // Clean subdirectory: remove leading/trailing slashes
                        let subdirectory = (subdirectoryWidget.value || "").trim();
                        subdirectory = subdirectory.replace(/^\/+|\/+$/g, '');

                        // Clear current preview immediately
                        node.previewImage = null;

                        // Create image preview
                        const img = new Image();
                        img.onload = () => {
                            // Only set preview after successful load
                            if (img.complete && img.naturalWidth > 0) {
                                node.previewImage = img;
                                app.graph.setDirtyCanvas(true, true);
                            } else {
                                console.error("Image onload fired but image not actually loaded");
                            }
                        };
                        img.onerror = () => {
                            console.error("Failed to load image from preview URL");
                            node.previewImage = null;
                            app.graph.setDirtyCanvas(true, true);
                        };

                        // Build API URL for image with preview optimization
                        const previewParams = app.getPreviewFormatParam ? app.getPreviewFormatParam() : '';
                        img.src = `/view?filename=${encodeURIComponent(filename)}&type=${directory}&subfolder=${encodeURIComponent(subdirectory)}${previewParams}&preview=true&channel=rgba&rand=${Math.random()}`;
                    };

                    // Function to update image list
                    const updateImageList = async (preserveCurrentValue = false) => {
                        const directory = directoryWidget.value;
                        // Clean subdirectory: remove leading/trailing slashes
                        let subdirectory = (subdirectoryWidget.value || "").trim();
                        subdirectory = subdirectory.replace(/^\/+|\/+$/g, '');
                        const prevValue = imageWidget.value;

                        try {
                            const response = await fetch(`/xjnodes/list_images?directory=${encodeURIComponent(directory)}&subdirectory=${encodeURIComponent(subdirectory)}`);
                            if (response.ok) {
                                const data = await response.json();

                                // Update image widget options
                                if (data.images.length > 0) {
                                    imageWidget.options.values = data.images;

                                    // Determine which image to select
                                    if (preserveCurrentValue) {
                                        // When preserving, check if current value exists in list
                                        if (prevValue && data.images.includes(prevValue)) {
                                            imageWidget.value = prevValue;
                                        } else {
                                            // Current image not in list, reset to first
                                            imageWidget.value = data.images[0];
                                        }
                                    } else {
                                        // Normal behavior: try to keep previous value, otherwise use first
                                        if (prevValue && data.images.includes(prevValue)) {
                                            imageWidget.value = prevValue;
                                        } else {
                                            imageWidget.value = data.images[0];
                                        }
                                    }
                                    // Update image preview with valid image
                                    updateImagePreview(imageWidget.value);
                                } else {
                                    // No images available - clear everything
                                    imageWidget.options.values = [];
                                    imageWidget.value = "";

                                    // Clear preview - we handle this ourselves with custom drawing
                                    node.previewImage = null;
                                }

                                // Force node to update its display
                                node.setDirtyCanvas(true, true);
                            } else {
                                console.error("API request failed:", response.status, response.statusText);
                            }
                        } catch (error) {
                            console.error("Error fetching image list:", error);
                        }
                    };

                    // Dummy function to get actual values from web page
                    const dummy = async () => {
                        // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
                    };

                    // Initial update - do this BEFORE setting up callbacks to avoid triggering them during load
                    await dummy(); // this will cause the widgets to obtain the actual value from web page.

                    // If we have a saved value from onConfigure, restore it before calling updateImageList
                    if (node._savedImageValue) {
                        imageWidget.value = node._savedImageValue;
                    }

                    // Call updateImageList with preserveCurrentValue=true to maintain the loaded value
                    await updateImageList(true);

                    // NOW hook into callbacks AFTER initial load is complete
                    // This prevents callbacks from firing during workflow deserialization

                    // Hook into directory widget change - preserve original callback
                    const originalDirectoryCallback = directoryWidget.callback;
                    directoryWidget.callback = function(value) {
                        if (originalDirectoryCallback) {
                            originalDirectoryCallback.apply(this, arguments);
                        }
                        updateImageList();
                    };

                    // Hook into subdirectory widget change - preserve original callback
                    const originalSubdirectoryCallback = subdirectoryWidget.callback;
                    subdirectoryWidget.callback = function(value) {
                        if (originalSubdirectoryCallback) {
                            originalSubdirectoryCallback.apply(this, arguments);
                        }
                        updateImageList();
                    };

                    // Hook into image widget change to update preview
                    const originalImageCallback = imageWidget.callback;
                    imageWidget.callback = function(value) {
                        if (originalImageCallback) {
                            originalImageCallback.apply(this, arguments);
                        }
                        updateImagePreview(value);
                    };

                    // Add Browse Subdirs button right after subdirectory widget (before image widget)
                    const subdirIndex = node.widgets.findIndex(w => w.name === "subdirectory");
                    const browseSubdirsButton = node.addWidget("button", "📁 Browse Subdirs", null, (value, widget, node, pos, event) => {
                        // Pass click event to help with positioning
                        showSubdirectoryDropdown(subdirectoryWidget.value, event);
                    });

                    // Make button not serialize (buttons shouldn't be saved to workflow)
                    browseSubdirsButton.serialize = false;

                    // Move the button to right after subdirectory widget
                    if (subdirIndex !== -1) {
                        // Remove button from end
                        const buttonWidget = node.widgets.pop();
                        // Insert it after subdirectory (at subdirIndex + 1)
                        node.widgets.splice(subdirIndex + 1, 0, buttonWidget);
                    }

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

                    // Make button not serialize (buttons shouldn't be saved to workflow)
                    navigatorButton.serialize = false;

                    // No need to override serialize - let ComfyUI save the sparse array
                    // We'll handle it correctly on the loading side in onConfigure

                    // Style the navigator button
                    navigatorButton.options.y = 8;

                    // Set initial node size to include space for preview image
                    // Default ComfyUI node width is around 210-320, we'll use a comfortable default
                    // Height should accommodate: title bar (~30px) + widgets (~75px for 3 widgets + button) + preview area (~200px) + margins
                    const defaultWidth = 320;
                    const defaultHeight = 340; // Enough space for widgets + preview

                    if (!node.size || node.size[0] < defaultWidth || node.size[1] < defaultHeight) {
                        node.size = [defaultWidth, defaultHeight];
                    }

                    // Save the updateImageList method for refreshing later
                    if (!registeredNodes[nodeData.name]) {
                        registeredNodes[nodeData.name] = [];
                    }
                    registeredNodes[nodeData.name].push(updateImageList);
                }

                return result;
            };

            // onConfigure is called when a node is loaded from a workflow
            // This is where we can access the saved widgets_values
            nodeType.prototype.onConfigure = function(info) {
                // Call original onConfigure if it exists
                if (originalOnConfigure) {
                    originalOnConfigure.apply(this, arguments);
                }

                // WORKAROUND: ComfyUI creates sparse arrays when widgets have serialize: false
                // At save time widget order is: [directory, subdirectory, browseSubdirsButton, image, browseImagesButton]
                // Sparse array is saved as: ['input', '', null, 'image.png', null]
                //
                // BUT: onConfigure is called BEFORE onNodeCreated adds the buttons!
                // So at this point widgets are: [directory, subdirectory, image]
                // We need to map from the current index to the saved index

                const imageWidget = this.widgets.find(w => w.name === "image");
                if (imageWidget && info?.widgets_values) {
                    // At this point (onConfigure), the widget order is the original from Python
                    // Original order: [directory, subdirectory, image]
                    // But the saved widgets_values was created with buttons inserted
                    // Saved order was: [directory, subdirectory, browseSubdirsButton, image, browseImagesButton]

                    // The saved image value is at index 3 (after directory, subdirectory, button)
                    // We look for the first non-null value after index 2 (after directory and subdirectory)
                    let savedImageValue = null;

                    // Skip past directory (0) and subdirectory (1), then find next non-null value
                    for (let i = 2; i < info.widgets_values.length; i++) {
                        if (info.widgets_values[i] !== null && info.widgets_values[i] !== undefined) {
                            savedImageValue = info.widgets_values[i];
                            break;
                        }
                    }

                    if (savedImageValue) {
                        // Store this on the node so onNodeCreated can use it
                        this._savedImageValue = savedImageValue;

                        // Add to options if not already there
                        if (!imageWidget.options.values.includes(savedImageValue)) {
                            imageWidget.options.values.push(savedImageValue);
                        }

                        // Set the value
                        imageWidget.value = savedImageValue;
                    }
                }
            };

            // Refresh existing nodes when node is re-registered (on refresh)
            if (registeredNodes[nodeData.name]) {
                for (const updateImageList of registeredNodes[nodeData.name]) {
                    await updateImageList(true); // Preserve current values on refresh
                }
            }
        }

        // Handle XJImagePairCompare node
        if (nodeData.name === "XJImagePairCompare") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            const originalOnExecuted = nodeType.prototype.onExecuted;
            const originalOnMouseDown = nodeType.prototype.onMouseDown;
            const originalOnMouseMove = nodeType.prototype.onMouseMove;
            const originalOnMouseEnter = nodeType.prototype.onMouseEnter;
            const originalOnMouseLeave = nodeType.prototype.onMouseLeave;
            const originalClick = nodeType.prototype.click;

            nodeType.prototype.onNodeCreated = function () {
                const result = originalNodeCreated?.apply(this, arguments);

                // Add custom widget - ComfyUI will automatically call its draw method
                this.pairCompareWidget = new XJImagePairCompareWidget("xj_pair_compare", this);
                this.addCustomWidget(this.pairCompareWidget);

                // Set default comparer_mode property
                if (!this.properties["comparer_mode"]) {
                    this.properties["comparer_mode"] = "Slide";
                }

                // Set initial node size
                this.setSize([320, 300]);

                return result;
            };

            nodeType.prototype.onExecuted = function (message) {
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }

                // Try different possible locations for the image data
                let aImages = message?.ui?.a_images || message?.a_images || [];
                let bImages = message?.ui?.b_images || message?.b_images || [];

                // Convert image data to URLs and create pairs
                const previewParams = app.getPreviewFormatParam ? app.getPreviewFormatParam() : '';
                const randParam = app.getRandParam ? app.getRandParam() : '';
                const pairs = [];

                for (let i = 0; i < aImages.length && i < bImages.length; i++) {
                    const aData = aImages[i];
                    const bData = bImages[i];

                    const aUrl = `/view?filename=${encodeURIComponent(aData.filename)}&type=${aData.type}&subfolder=${encodeURIComponent(aData.subfolder || '')}${previewParams}${randParam}`;
                    const bUrl = `/view?filename=${encodeURIComponent(bData.filename)}&type=${bData.type}&subfolder=${encodeURIComponent(bData.subfolder || '')}${previewParams}${randParam}`;

                    pairs.push({
                        a_url: aUrl,
                        b_url: bUrl,
                    });
                }

                // Update widget value
                if (this.pairCompareWidget) {
                    this.pairCompareWidget.value = { pairs, selectedIndex: 0 };
                }

                this.setDirtyCanvas(true, true);
            };

            // Tab clicks are now handled by the widget's mouse() method
            // This matches the rgthree implementation pattern
            nodeType.prototype.onMouseDown = function (event, pos, canvas) {
                if (originalOnMouseDown) {
                    return originalOnMouseDown.apply(this, arguments);
                }
                return false;
            };

            nodeType.prototype.onMouseMove = function (event, pos, canvas) {
                if (this.pairCompareWidget) {
                    this.pairCompareWidget.onMouseMove(event, pos, this);
                    this.pairCompareWidget.setIsPointerDown(true);
                    this.setDirtyCanvas(true, false);
                }
                if (originalOnMouseMove) {
                    originalOnMouseMove.apply(this, arguments);
                }
            };

            nodeType.prototype.onMouseEnter = function (event) {
                if (this.pairCompareWidget) {
                    this.pairCompareWidget.onMouseEnter();
                }
                if (originalOnMouseEnter) {
                    originalOnMouseEnter.apply(this, arguments);
                }
            };

            nodeType.prototype.onMouseLeave = function (event) {
                if (this.pairCompareWidget) {
                    this.pairCompareWidget.onMouseLeave();
                    this.pairCompareWidget.setIsPointerDown(false);
                }
                if (originalOnMouseLeave) {
                    originalOnMouseLeave.apply(this, arguments);
                }
            };

            // Support for comparer_mode property
            nodeData.prototype = nodeType.prototype;
            nodeType.prototype["@comparer_mode"] = {
                type: "combo",
                values: ["Slide", "Click"],
            };

            // Add context menu for opening images
            const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (originalGetExtraMenuOptions) {
                    originalGetExtraMenuOptions.apply(this, arguments);
                }

                // Add "Open image" options for current pair
                if (this.pairCompareWidget && this.pairCompareWidget.pairCount > 0) {
                    const selectedIndex = this.pairCompareWidget.selectedPairIndex;
                    const pairs = this.pairCompareWidget._value.pairs;

                    if (pairs && pairs[selectedIndex]) {
                        const pair = pairs[selectedIndex];

                        options.unshift(
                            {
                                content: "Open Image A",
                                callback: () => {
                                    window.open(pair.a_url, "_blank");
                                }
                            },
                            {
                                content: "Open Image B",
                                callback: () => {
                                    window.open(pair.b_url, "_blank");
                                }
                            }
                        );
                    }
                }
            };
        }

        // Handle XJConditionalLoraLoader node
        if (nodeData.name === "XJConditionalLoraLoader") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = async function () {
                const result = originalNodeCreated?.apply(this, arguments);

                this.loraCounter = 0;
                this.loraWidgets = [];
                this.serialize_widgets = true;
                this.cachedLoraList = ["None"]; // Cache to avoid repeated API calls

                // Fetch loras immediately on node creation
                await this.fetchLoraList();

                // Add initial "Add Lora" button
                this.addLoraButton = this.addWidget("button", "➕ Add Lora", null, () => {
                    this.addLoraSet();
                });

                // Don't serialize the button
                this.addLoraButton.serialize = false;

                // Restore saved state if loading from workflow
                if (this._savedLoraWidgetValues) {
                    // widgets_values format: [enabled, lora1_dict, lora2_dict, ...]
                    // Skip first value (enabled toggle), then process dict values
                    for (let i = 1; i < this._savedLoraWidgetValues.length; i++) {
                        const loraData = this._savedLoraWidgetValues[i];

                        // Check if this is a dict with lora and strength
                        if (loraData && typeof loraData === 'object' && 'lora' in loraData && 'strength' in loraData) {
                            // Add a new lora set (skipResize=true to preserve saved node size)
                            const widgetSet = this.addLoraSet(true);

                            // Restore the hidden data widget (this is what gets sent to backend)
                            widgetSet.data.value = loraData;

                            // Sync the display widgets from the data
                            widgetSet.lora.value = loraData.lora;
                            widgetSet.strength.value = loraData.strength;
                        }
                    }

                    // Clear saved state after restoration
                    delete this._savedLoraWidgetValues;
                }

                // Don't force size - let ComfyUI preserve it from saved workflow or auto-calculate

                return result;
            };

            // Method to fetch lora list from API
            nodeType.prototype.fetchLoraList = async function() {
                try {
                    const resp = await fetch("/xjnodes/get_loras");
                    if (resp.ok) {
                        const data = await resp.json();
                        if (data.success && data.loras) {
                            this.cachedLoraList = ["None", ...data.loras];
                        }
                    }
                } catch (e) {
                    console.error("[XJConditionalLoraLoader] Failed to fetch lora list:", e);
                }
            };

            // Method to get lora list (returns cached list)
            nodeType.prototype.getLoraList = function() {
                return this.cachedLoraList || ["None"];
            };

            // Method to refresh lora list asynchronously
            nodeType.prototype.refreshLoraList = async function() {
                await this.fetchLoraList();

                // Update all existing lora widgets with new list
                const loraList = this.getLoraList();
                for (const widgetSet of this.loraWidgets) {
                    const currentValue = widgetSet.lora.value;
                    widgetSet.lora.options.values = loraList;
                    // Restore value if it still exists
                    if (loraList.includes(currentValue)) {
                        widgetSet.lora.value = currentValue;
                    }
                }
                this.setDirtyCanvas(true, true);
            };

            // Method to add a set of lora widgets
            nodeType.prototype.addLoraSet = function(skipResize = false) {
                this.loraCounter++;
                const loraNum = this.loraCounter;
                const loraKey = `lora_${loraNum}`;

                // Get list of available loras
                const loraList = this.getLoraList();

                // Create a hidden widget that stores the dict value
                const dataWidget = this.addWidget("combo", loraKey, { lora: "None", strength: 1.0 }, null, { values: ["hidden"] });
                dataWidget.type = "converted-widget"; // Hide from UI
                dataWidget.computeSize = () => [0, -4]; // Make it invisible

                // Create visible UI widgets
                const loraWidget = this.addWidget("combo", `${loraKey}_display`, "None", (value) => {
                    // Update the hidden data widget when lora changes
                    dataWidget.value = {
                        lora: value,
                        strength: strengthWidget.value
                    };
                    this.setDirtyCanvas(true);
                }, { values: loraList });
                loraWidget.label = `LoRA ${loraNum}`;
                loraWidget.serialize = false; // Don't serialize display widget

                const strengthWidget = this.addWidget("number", `${loraKey}_strength_display`, 1.0, (value) => {
                    // Update the hidden data widget when strength changes
                    dataWidget.value = {
                        lora: loraWidget.value,
                        strength: value
                    };
                    this.setDirtyCanvas(true);
                }, { min: -10.0, max: 10.0, step: 0.01, precision: 2 });
                strengthWidget.label = `Strength ${loraNum}`;
                strengthWidget.serialize = false; // Don't serialize display widget

                const removeButton = this.addWidget("button", `🗑️ Remove`, null, () => {
                    this.removeLoraSet(loraNum);
                });
                removeButton.serialize = false;

                // Store widget references
                const widgetSet = {
                    id: loraNum,
                    key: loraKey,
                    data: dataWidget,      // Hidden widget with dict value (serialized, sent to backend)
                    lora: loraWidget,      // Display widget for lora selection
                    strength: strengthWidget, // Display widget for strength
                    remove: removeButton
                };

                this.loraWidgets.push(widgetSet);

                // Move "Add Lora" button to the end
                const buttonIndex = this.widgets.indexOf(this.addLoraButton);
                if (buttonIndex !== -1) {
                    this.widgets.splice(buttonIndex, 1);
                    this.widgets.push(this.addLoraButton);
                }

                // Resize node to fit all widgets (only if not restoring from saved workflow)
                if (!skipResize) {
                    const newHeight = Math.max(120, 50 + this.widgets.length * 25);
                    this.setSize([320, newHeight]);
                }

                return widgetSet;
            };

            // Method to remove a lora set
            nodeType.prototype.removeLoraSet = function(loraNum) {
                const widgetSet = this.loraWidgets.find(w => w.id === loraNum);
                if (!widgetSet) return;

                // Remove all widgets in this set (including hidden data widget)
                const widgetsToRemove = [
                    widgetSet.data,
                    widgetSet.lora,
                    widgetSet.strength,
                    widgetSet.remove
                ];

                for (const widget of widgetsToRemove) {
                    const index = this.widgets.indexOf(widget);
                    if (index !== -1) {
                        this.widgets.splice(index, 1);
                    }
                }

                // Remove from loraWidgets array
                const setIndex = this.loraWidgets.indexOf(widgetSet);
                if (setIndex !== -1) {
                    this.loraWidgets.splice(setIndex, 1);
                }

                // Resize node
                const newHeight = Math.max(120, 50 + this.widgets.length * 25);
                this.setSize([320, newHeight]);

                this.setDirtyCanvas(true, true);
            };

            // Override getExtraMenuOptions to add refresh loras option
            const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (originalGetExtraMenuOptions) {
                    originalGetExtraMenuOptions.apply(this, arguments);
                }

                options.unshift({
                    content: "Refresh Loras",
                    callback: () => {
                        this.refreshLoraList();
                    }
                });
            };

            // Handle refresh from ComfyUI's refresh button
            const originalRefreshComboInNode = nodeType.prototype.refreshComboInNode;
            nodeType.prototype.refreshComboInNode = function() {
                if (originalRefreshComboInNode) {
                    originalRefreshComboInNode.apply(this, arguments);
                }
                this.refreshLoraList();
            };

            // Handle loading from saved workflows
            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (originalOnConfigure) {
                    originalOnConfigure.apply(this, arguments);
                }

                // Save widget values for restoration in onNodeCreated
                // onConfigure is called BEFORE onNodeCreated, so we can't restore state here
                if (info.widgets_values) {
                    this._savedLoraWidgetValues = info.widgets_values;
                }
            };
        }
    }
});