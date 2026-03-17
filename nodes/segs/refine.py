"""
Mask refinement nodes using k-means clustering and morphological operations.
Helps grow under-segmented masks from SAM3 or other segmentation models.
"""

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from scipy import ndimage

from .core import SEG, create_segs, get_segs_shape, get_segs_list


class XJMaskRefineKMeans:
    """
    Refine mask boundary using k-means clustering on image colors.
    Grows mask to include similar colored pixels at the boundary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "n_clusters": ("INT", {"default": 3, "min": 2, "max": 10, "step": 1, "tooltip": "Number of k-means clusters to use for color segmentation. More clusters can capture more color variation but may be slower."}),
                "grow_pixels": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1, "tooltip": "Maximum pixels to grow the mask. Use with similarity_threshold to control how far the growth extends."}),
                "similarity_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold for including similar clusters (0-1). Higher = more strict. Use to include clusters that are close in color to the original mask, for smoother boundaries."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refined_mask",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "refine"

    def refine(self, image, mask, n_clusters, grow_pixels, similarity_threshold):
        """
        Refine mask using k-means clustering.

        Args:
            image: Input image (B, H, W, C) or (H, W, C)
            mask: Input mask (B, H, W) or (H, W)
            n_clusters: Number of k-means clusters
            grow_pixels: Maximum pixels to grow the mask
            similarity_threshold: Threshold for including similar clusters (0-1)

        Returns:
            Refined mask
        """
        # Handle batch dimension
        if len(image.shape) == 4:
            # Process each batch item
            results = []
            for i in range(image.shape[0]):
                img = image[i]  # (H, W, C)
                msk = mask[i] if len(mask.shape) == 3 else mask  # (H, W)
                refined = self._refine_single(img, msk, n_clusters, grow_pixels, similarity_threshold)
                results.append(refined)
            return (torch.stack(results),)
        else:
            refined = self._refine_single(image, mask, n_clusters, grow_pixels, similarity_threshold)
            return (refined.unsqueeze(0),)

    def _refine_single(self, image, mask, n_clusters, grow_pixels, similarity_threshold):
        """Refine a single image/mask pair with smooth edge growth."""
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
        else:
            img_np = np.array(image)

        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Ensure image is (H, W, C)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[0] == 3 and img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Ensure mask is (H, W)
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)

        # Normalize image to [0, 1]
        if img_np.max() > 1.5:
            img_np = img_np / 255.0

        binary_mask = mask_np > 0.5

        # Create dilated mask for maximum growth region
        if grow_pixels > 0:
            dilated_mask = ndimage.binary_dilation(binary_mask, iterations=grow_pixels)
        else:
            dilated_mask = binary_mask

        # Get boundary region (dilated - original)
        boundary = dilated_mask & ~binary_mask

        if not boundary.any():
            return torch.from_numpy(mask_np.astype(np.float32))

        # Get pixels in the dilated region for clustering
        roi_pixels = img_np[dilated_mask]

        if len(roi_pixels) < n_clusters:
            # Not enough pixels for clustering, use smooth morphological growth
            return self._smooth_grow(binary_mask, grow_pixels)

        # Get pixels in original mask for reference
        mask_pixels = img_np[binary_mask]
        if len(mask_pixels) == 0:
            return torch.from_numpy(mask_np.astype(np.float32))

        # Compute average color of the original mask
        object_color = np.mean(mask_pixels, axis=0)

        # Perform k-means clustering on ROI
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(roi_pixels)

        # Find which cluster center is closest to the object's average color
        distances_to_object = np.linalg.norm(kmeans.cluster_centers_ - object_color, axis=1)
        object_cluster_idx = np.argmin(distances_to_object)

        # Create a color similarity mask for the entire dilated region
        all_roi_pixels = img_np[dilated_mask]
        all_labels = kmeans.predict(all_roi_pixels)

        # Build color mask: True where pixels belong to object cluster
        color_mask = np.zeros(dilated_mask.shape, dtype=bool)
        color_mask[dilated_mask] = (all_labels == object_cluster_idx)

        # Additional color similarity check for smoother boundaries
        if similarity_threshold > 0:
            # Compute color distance for all pixels in dilated region
            color_distances = np.linalg.norm(img_np - object_color, axis=2)
            max_dist = np.sqrt(3.0)
            similarity_mask = (1.0 - color_distances / max_dist) >= similarity_threshold
            color_mask = color_mask & similarity_mask

        # Smooth growth: use morphological operations on the color-based mask
        # First, intersect with dilated region to stay within bounds
        grown_mask = binary_mask | (color_mask & boundary)

        # Aggressive smoothing to remove spikes:
        # 1. Opening: removes small protrusions (spikes)
        grown_mask = ndimage.binary_opening(grown_mask, iterations=2)

        # 2. Closing: fills small holes
        grown_mask = ndimage.binary_closing(grown_mask, iterations=3)

        # 3. Second opening to clean up after closing
        grown_mask = ndimage.binary_opening(grown_mask, iterations=1)

        # 4. Distance transform for smooth boundaries
        # This creates a gradient from the mask center to edges
        distance = ndimage.distance_transform_edt(grown_mask)
        distance_inv = ndimage.distance_transform_edt(~grown_mask)

        # Combine distances for smooth transition
        smooth_boundary = distance - distance_inv

        # Apply Gaussian smoothing for extra smoothness
        smoothed = ndimage.gaussian_filter(smooth_boundary, sigma=2.0)

        # Threshold to get final binary mask (0 crossing gives smooth boundary)
        refined_mask = smoothed > 0

        return torch.from_numpy(refined_mask.astype(np.float32))

    def _smooth_grow(self, binary_mask, grow_pixels):
        """Smooth morphological growth when clustering fails."""
        if grow_pixels > 0:
            grown = ndimage.binary_dilation(binary_mask, iterations=grow_pixels)
        else:
            grown = binary_mask

        # Smooth with Gaussian filter
        float_mask = grown.astype(np.float32)
        smoothed = ndimage.gaussian_filter(float_mask, sigma=1.5)
        refined = smoothed > 0.5

        return torch.from_numpy(refined.astype(np.float32))


class XJMaskRefineMorph:
    """
    Refine mask boundary using morphological operations.
    Grows mask using dilation and smooths edges.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "grow_pixels": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1, "tooltip": "Number of pixels to dilate/grow. Use with edge_threshold and color_threshold to control growth."}),
                "smooth_pixels": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1, "tooltip": "Sigma for Gaussian smoothing. Higher values create smoother but less detailed edges."}),
                "edge_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold for edge-based refinement (0-1). Lower values are more sensitive to edges. Use with use_edge_detection to control growth at edges."}),
                "use_edge_detection": ("BOOLEAN", {"default": True, "tooltip": "Whether to use edge detection to limit growth. Disable for similar colors or when edge detection is too aggressive."}),
                "color_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "If > 0, stop growth when pixel color differs from mask average by more than this threshold (0-1). Use instead of edge detection for similar colors."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refined_mask",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "refine"

    def refine(self, image, mask, grow_pixels, smooth_pixels, edge_threshold, use_edge_detection=True, color_threshold=0.0):
        """
        Refine mask using morphological operations.

        Args:
            image: Input image (B, H, W, C) or (H, W, C)
            mask: Input mask (B, H, W) or (H, W)
            grow_pixels: Number of pixels to dilate/grow
            smooth_pixels: Sigma for Gaussian smoothing
            edge_threshold: Threshold for edge-based refinement
            use_edge_detection: Whether to use edge detection (disable for similar colors)
            color_threshold: If > 0, stop growth when pixel color differs from mask average by more than this (0-1)

        Returns:
            Refined mask
        """
        # Handle batch dimension
        if len(image.shape) == 4:
            results = []
            for i in range(image.shape[0]):
                img = image[i]
                msk = mask[i] if len(mask.shape) == 3 else mask
                refined = self._refine_single(img, msk, grow_pixels, smooth_pixels, edge_threshold, use_edge_detection, color_threshold)
                results.append(refined)
            return (torch.stack(results),)
        else:
            refined = self._refine_single(image, mask, grow_pixels, smooth_pixels, edge_threshold, use_edge_detection, color_threshold)
            return (refined.unsqueeze(0),)

    def _refine_single(self, image, mask, grow_pixels, smooth_pixels, edge_threshold, use_edge_detection=True, color_threshold=0.0):
        """Refine a single image/mask pair with edge-aware or color-aware growth."""
        # Convert to numpy
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Ensure mask is (H, W)
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)

        # Binary mask
        binary_mask = mask_np > 0.5

        # Handle image
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
        else:
            img_np = np.array(image)

        # Ensure image is (H, W, C)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[0] == 3 and img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        if img_np.max() > 1.5:
            img_np = img_np / 255.0

        # Compute object color for color-aware growth
        object_color = None
        if color_threshold > 0:
            mask_pixels = img_np[binary_mask]
            if len(mask_pixels) > 0:
                object_color = np.mean(mask_pixels, axis=0)
                print(f"[XJMaskRefineMorph] Object color: {object_color}, color_threshold={color_threshold}")

        if use_edge_detection and edge_threshold > 0 and color_threshold == 0:
            print(f"[XJMaskRefineMorph] Using edge detection with threshold={edge_threshold}")
            # Compute image edges using gradient magnitude
            gray = np.mean(img_np, axis=2)
            sobel_x = ndimage.sobel(gray, axis=1)
            sobel_y = ndimage.sobel(gray, axis=0)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            # Normalize edges using percentile for better threshold behavior
            edge_max = np.percentile(edge_magnitude, 95)  # Use 95th percentile as max
            if edge_max > 0:
                edge_magnitude = edge_magnitude / edge_max
            edge_magnitude = np.clip(edge_magnitude, 0, 1)

            # Create edge mask: high values = strong edges
            # We want to STOP growth at strong edges, so invert: low edge = can grow
            edge_mask = edge_magnitude < edge_threshold

            # Iterative growth with edge checking (more controlled than one big dilation)
            current_mask = binary_mask.copy()
            for _ in range(grow_pixels):
                # Dilate by 1 pixel
                dilated = ndimage.binary_dilation(current_mask, iterations=1)
                # New pixels added this iteration
                new_pixels = dilated & ~current_mask
                # Only keep new pixels that are NOT on strong edges
                allowed_new = new_pixels & edge_mask
                # Update mask
                current_mask = current_mask | allowed_new
            print(f"[XJMaskRefineMorph] Edge mode: original pixels={binary_mask.sum()}, final={current_mask.sum()}")
        elif color_threshold > 0 and object_color is not None:
            print(f"[XJMaskRefineMorph] Using color-aware growth, color_threshold={color_threshold}")
            # Compute color distance for all pixels
            color_distances = np.linalg.norm(img_np - object_color, axis=2)
            max_dist = np.sqrt(3.0)  # Max distance in RGB space (0-1)
            color_similarity = color_distances / max_dist  # Normalize to 0-1
            color_mask = color_similarity < color_threshold  # True = similar enough to object

            # Iterative growth with color checking
            current_mask = binary_mask.copy()
            for _ in range(grow_pixels):
                # Dilate by 1 pixel
                dilated = ndimage.binary_dilation(current_mask, iterations=1)
                # New pixels added this iteration
                new_pixels = dilated & ~current_mask
                # Only keep new pixels that are color-similar to the object
                allowed_new = new_pixels & color_mask
                # Update mask
                current_mask = current_mask | allowed_new
                # Stop if no new pixels added
                if not allowed_new.any():
                    break
            print(f"[XJMaskRefineMorph] Color mode: original pixels={binary_mask.sum()}, final={current_mask.sum()}")
        else:
            print(f"[XJMaskRefineMorph] Pure dilation mode, grow_pixels={grow_pixels}")
            # Pure morphological growth without edge detection
            # Use a single dilation with the full grow_pixels for efficiency
            if grow_pixels > 0:
                current_mask = ndimage.binary_dilation(binary_mask, iterations=grow_pixels)
            else:
                current_mask = binary_mask
            print(f"[XJMaskRefineMorph] Pure dilation: original pixels={binary_mask.sum()}, final={current_mask.sum()}")

        # Apply morphological closing to smooth the result
        refined_mask = ndimage.binary_closing(current_mask, iterations=2)

        # Step 3: Smooth the mask boundary
        if smooth_pixels > 0:
            float_mask = refined_mask.astype(np.float32)
            smoothed = ndimage.gaussian_filter(float_mask, sigma=smooth_pixels)
            refined_mask = smoothed > 0.5

        return torch.from_numpy(refined_mask.astype(np.float32))


class XJSegsRefineKMeans:
    """
    Refine SEGS masks using k-means clustering on image colors.
    Processes each segment individually.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "image": ("IMAGE",),
                "n_clusters": ("INT", {"default": 3, "min": 2, "max": 10, "step": 1, "tooltip": "Number of k-means clusters to use for color segmentation. More clusters can capture more color variation but may be slower."}),
                "grow_pixels": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1, "tooltip": "Maximum pixels to grow the mask. Use with similarity_threshold to control how far the growth extends."}),
                "similarity_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold for including similar clusters (0-1). Higher = more strict. Use to include clusters that are close in color to the original mask, for smoother boundaries."}),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("refined_segs",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "refine"

    def refine(self, segs, image, n_clusters, grow_pixels, similarity_threshold):
        """
        Refine SEGS using k-means clustering.

        Args:
            segs: SEGS tuple ((height, width), [SEG, ...])
            image: Full image (B, H, W, C) - uses first batch if B > 1
            n_clusters: Number of k-means clusters
            grow_pixels: Maximum pixels to grow each mask
            similarity_threshold: Threshold for including similar clusters

        Returns:
            Refined SEGS
        """
        shape = get_segs_shape(segs)
        seg_list = get_segs_list(segs)

        if not seg_list:
            return (segs,)

        # Use first image from batch
        img = image[0] if len(image.shape) == 4 else image

        refined_segments = []

        for seg in seg_list:
            x1, y1, x2, y2 = seg.crop_region

            # Crop image region
            crop_img = img[y1:y2, x1:x2]

            # Get mask
            mask = seg.cropped_mask
            if isinstance(mask, Image.Image):
                mask = np.array(mask).astype(np.float32) / 255.0
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            # Ensure mask is (H, W)
            if len(mask.shape) == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                else:
                    mask = mask.squeeze(-1)

            # Refine using k-means
            kmeans_refine = XJMaskRefineKMeans()
            refined_mask = kmeans_refine._refine_single(
                crop_img, mask, n_clusters, grow_pixels, similarity_threshold
            )

            # Convert to numpy array for SEGS compatibility with Impact Pack nodes
            if isinstance(refined_mask, torch.Tensor):
                refined_mask = refined_mask.cpu().numpy()

            # Ensure mask has shape (1, H, W) for SEGS compatibility
            if len(refined_mask.shape) == 2:
                refined_mask = refined_mask[np.newaxis, ...]

            # Create new SEG with refined mask
            refined_seg = SEG(
                cropped_image=seg.cropped_image,
                cropped_mask=refined_mask,
                confidence=seg.confidence,
                crop_region=seg.crop_region,
                bbox=seg.bbox,
                label=seg.label,
                control_net_wrapper=seg.control_net_wrapper,
            )
            refined_segments.append(refined_seg)

        return (create_segs(shape, refined_segments),)


class XJSegsRefineMorph:
    """
    Refine SEGS masks using morphological operations.
    Processes each segment individually with edge-aware growing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "image": ("IMAGE",),
                "grow_pixels": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1, "tooltip": "Number of pixels to dilate/grow. Use with edge_threshold and color_threshold to control growth."}),
                "smooth_pixels": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1, "tooltip": "Sigma for Gaussian smoothing. Higher values create smoother but less detailed edges."}),
                "edge_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold for edge-based refinement (0-1). Lower values are more sensitive to edges. Use with use_edge_detection to control growth at edges."}),
                "use_edge_detection": ("BOOLEAN", {"default": True, "tooltip": "Whether to use edge detection to limit growth. Disable for similar colors or when edge detection is too aggressive."}),
                "color_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "If > 0, stop growth when pixel color differs from mask average by more than this threshold (0-1). Use instead of edge detection for similar colors."}),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("refined_segs",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "refine"

    def refine(self, segs, image, grow_pixels, smooth_pixels, edge_threshold, use_edge_detection=True, color_threshold=0.0):
        """
        Refine SEGS using morphological operations.

        Args:
            segs: SEGS tuple ((height, width), [SEG, ...])
            image: Full image (B, H, W, C) - uses first batch if B > 1
            grow_pixels: Number of pixels to dilate/grow
            smooth_pixels: Sigma for Gaussian smoothing
            edge_threshold: Threshold for edge-based refinement
            use_edge_detection: Whether to use edge detection (disable for similar colors)
            color_threshold: If > 0, stop growth when pixel color differs from mask average

        Returns:
            Refined SEGS
        """
        shape = get_segs_shape(segs)
        seg_list = get_segs_list(segs)

        if not seg_list:
            return (segs,)

        # Use first image from batch
        img = image[0] if len(image.shape) == 4 else image

        refined_segments = []

        for seg in seg_list:
            x1, y1, x2, y2 = seg.crop_region
            crop_w, crop_h = x2 - x1, y2 - y1

            # Crop image region
            crop_img = img[y1:y2, x1:x2]

            # Get mask
            mask = seg.cropped_mask
            if isinstance(mask, Image.Image):
                mask = np.array(mask).astype(np.float32) / 255.0
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            # Ensure mask is (H, W)
            if len(mask.shape) == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                else:
                    mask = mask.squeeze(-1)

            print(f"[XJSegsRefineMorph] Segment crop_region=({x1},{y1},{x2},{y2}) size={crop_w}x{crop_h}, mask shape={mask.shape}")

            # Refine using morphological operations
            morph_refine = XJMaskRefineMorph()
            refined_mask = morph_refine._refine_single(
                crop_img, mask, grow_pixels, smooth_pixels, edge_threshold, use_edge_detection, color_threshold
            )

            # Convert to numpy array for SEGS compatibility with Impact Pack nodes
            if isinstance(refined_mask, torch.Tensor):
                refined_mask = refined_mask.cpu().numpy()

            # Ensure mask has shape (1, H, W) for SEGS compatibility
            if len(refined_mask.shape) == 2:
                refined_mask = refined_mask[np.newaxis, ...]

            print(f"[XJSegsRefineMorph] Refined mask shape={refined_mask.shape}, sum={refined_mask.sum():.0f}")

            # Create new SEG with refined mask
            refined_seg = SEG(
                cropped_image=seg.cropped_image,
                cropped_mask=refined_mask,
                confidence=seg.confidence,
                crop_region=seg.crop_region,
                bbox=seg.bbox,
                label=seg.label,
                control_net_wrapper=seg.control_net_wrapper,
            )
            refined_segments.append(refined_seg)

        return (create_segs(shape, refined_segments),)


class XJSegsDilateMaskExpand:
    """
    Dilate mask in SEGS with expanded crop region to preserve full mask shape.
    Unlike standard dilation that is limited by crop_region boundaries,
    this node expands the crop_region to accommodate the full dilated mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "expand_buffer": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "Extra pixels to expand beyond dilation amount. 0 = expand exactly enough for dilation."}),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("expanded_segs",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "dilate_expand"

    def dilate_expand(self, segs, dilation, expand_buffer=0):
        """
        Dilate masks in SEGS while expanding crop_region to fit the full mask.

        Args:
            segs: SEGS tuple ((height, width), [SEG, ...])
            dilation: Number of pixels to dilate (positive) or erode (negative)
            expand_buffer: Additional pixels to expand beyond dilation amount

        Returns:
            SEGS with dilated masks and expanded crop regions
        """
        shape = get_segs_shape(segs)
        seg_list = get_segs_list(segs)

        if not seg_list:
            return (segs,)

        new_segments = []

        for seg in seg_list:
            # Get original crop region and dimensions
            cx1, cy1, cx2, cy2 = seg.crop_region
            crop_w, crop_h = cx2 - cx1, cy2 - cy1

            # Get original bbox
            bx1, by1, bx2, by2 = seg.bbox

            # Calculate expansion needed
            # We need to expand by at least abs(dilation) + expand_buffer
            expand = abs(dilation) + expand_buffer

            # Get the original mask
            mask = seg.cropped_mask
            if isinstance(mask, Image.Image):
                mask = np.array(mask).astype(np.float32) / 255.0
            elif isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            # Ensure mask is 2D for processing
            orig_mask_shape = mask.shape
            if mask.ndim == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                else:
                    mask = mask.squeeze(-1)

            # Pad the mask to accommodate dilation
            # We pad on all sides by the expand amount
            if dilation != 0:
                padded_mask = np.pad(mask, ((expand, expand), (expand, expand)), mode='constant', constant_values=0)
            else:
                padded_mask = mask.copy()

            # Apply dilation/erosion using scipy.ndimage with circular structuring element
            if dilation != 0:
                binary_mask = padded_mask > 0.5
                # Create circular structuring element for natural shape preservation
                # Using a disk shape (approximated by distance transform)
                from scipy.ndimage import distance_transform_edt
                import numpy as np

                # Create a disk-shaped footprint
                def disk_footprint(radius):
                    """Create a circular footprint with given radius."""
                    size = int(2 * radius + 1)
                    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                    return x**2 + y**2 <= radius**2

                footprint = disk_footprint(abs(dilation))

                if dilation > 0:
                    # Dilate using binary_dilation with circular footprint
                    dilated_binary = ndimage.binary_dilation(binary_mask, structure=footprint)
                else:
                    # Erode using binary_erosion with circular footprint
                    dilated_binary = ndimage.binary_erosion(binary_mask, structure=footprint)
                dilated_mask = dilated_binary.astype(np.float32)
            else:
                dilated_mask = padded_mask

            # Calculate new crop region coordinates
            new_cx1 = max(0, cx1 - expand)
            new_cy1 = max(0, cy1 - expand)
            new_cx2 = min(shape[1], cx2 + expand)  # shape[1] is width
            new_cy2 = min(shape[0], cy2 + expand)  # shape[0] is height

            # Adjust mask if expansion was clamped at image boundaries
            actual_expand_left = cx1 - new_cx1
            actual_expand_top = cy1 - new_cy1
            actual_expand_right = new_cx2 - cx2
            actual_expand_bottom = new_cy2 - cy2

            # Trim the dilated mask if we hit image boundaries
            mask_h, mask_w = dilated_mask.shape
            trim_top = expand - actual_expand_top
            trim_bottom = mask_h - (expand - actual_expand_bottom)
            trim_left = expand - actual_expand_left
            trim_right = mask_w - (expand - actual_expand_right)

            dilated_mask = dilated_mask[trim_top:trim_bottom, trim_left:trim_right]

            # Calculate new bbox in the new crop region coordinate space
            # The bbox shifts by the expansion amount (with boundary clamping adjustments)
            new_bx1 = bx1 - cx1 + actual_expand_left
            new_by1 = by1 - cy1 + actual_expand_top
            new_bx2 = bx2 - cx1 + actual_expand_left
            new_by2 = by2 - cy1 + actual_expand_top

            # Ensure bbox is within new crop region
            new_crop_w = new_cx2 - new_cx1
            new_crop_h = new_cy2 - new_cy1
            new_bx1 = max(0, min(new_bx1, new_crop_w - 1))
            new_by1 = max(0, min(new_by1, new_crop_h - 1))
            new_bx2 = max(new_bx1 + 1, min(new_bx2, new_crop_w))
            new_by2 = max(new_by1 + 1, min(new_by2, new_crop_h))

            # Ensure mask has shape (1, H, W) for SEGS compatibility
            if len(dilated_mask.shape) == 2:
                dilated_mask = dilated_mask[np.newaxis, ...]

            # Create new cropped_image by expanding the original
            # We need to extract the expanded region from the original image
            cropped_image = seg.cropped_image
            if cropped_image is not None:
                if isinstance(cropped_image, torch.Tensor):
                    img = cropped_image.cpu().numpy()
                else:
                    img = np.array(cropped_image)

                # Handle different image formats (H, W, C) or (C, H, W)
                if img.ndim == 3:
                    if img.shape[0] == 3 or img.shape[0] == 4:  # (C, H, W) format
                        # Pad each channel
                        img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
                        padded_img = np.pad(img, ((expand, expand), (expand, expand), (0, 0)), mode='constant', constant_values=0)
                        # Apply trimming
                        padded_img = padded_img[trim_top:trim_bottom, trim_left:trim_right]
                        padded_img = np.transpose(padded_img, (2, 0, 1))  # Convert back to (C, H, W)
                    else:  # (H, W, C) format
                        padded_img = np.pad(img, ((expand, expand), (expand, expand), (0, 0)), mode='constant', constant_values=0)
                        padded_img = padded_img[trim_top:trim_bottom, trim_left:trim_right]

                    cropped_image = torch.from_numpy(padded_img) if isinstance(seg.cropped_image, torch.Tensor) else padded_img

            # Create new SEG with expanded dimensions
            new_seg = SEG(
                cropped_image=cropped_image,
                cropped_mask=dilated_mask,
                confidence=seg.confidence,
                crop_region=(new_cx1, new_cy1, new_cx2, new_cy2),
                bbox=(new_bx1, new_by1, new_bx2, new_by2),
                label=seg.label,
                control_net_wrapper=seg.control_net_wrapper,
            )
            new_segments.append(new_seg)

        return (create_segs(shape, new_segments),)


class XJMaskDilateCircular:
    """
    Dilate or erode a mask using a circular structuring element.
    Preserves natural rounded shapes instead of creating square corners.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("dilated_mask",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "dilate_circular"

    def dilate_circular(self, mask, dilation):
        """
        Dilate/erode mask using circular structuring element.

        Args:
            mask: Input mask tensor (B, H, W) or (H, W)
            dilation: Positive to dilate, negative to erode

        Returns:
            Dilated/eroded mask with preserved circular shapes
        """
        import torch

        # Handle batch dimension
        if len(mask.shape) == 3:
            results = []
            for i in range(mask.shape[0]):
                m = mask[i]
                result = self._dilate_single(m, dilation)
                results.append(result)
            return (torch.stack(results),)
        else:
            result = self._dilate_single(mask, dilation)
            return (result.unsqueeze(0),)

    def _dilate_single(self, mask, dilation):
        """Process a single mask."""
        import torch
        import numpy as np
        from scipy import ndimage

        if dilation == 0:
            return mask

        # Convert to numpy
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Ensure 2D
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()

        binary_mask = mask_np > 0.5

        # Create circular footprint
        radius = abs(dilation)
        size = int(2 * radius + 1)
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        footprint = x**2 + y**2 <= radius**2

        if dilation > 0:
            result = ndimage.binary_dilation(binary_mask, structure=footprint)
        else:
            result = ndimage.binary_erosion(binary_mask, structure=footprint)

        return torch.from_numpy(result.astype(np.float32))


NODE_CLASS_MAPPINGS = {
    "XJMaskRefineKMeans": XJMaskRefineKMeans,
    "XJMaskRefineMorph": XJMaskRefineMorph,
    "XJSegsRefineKMeans": XJSegsRefineKMeans,
    "XJSegsRefineMorph": XJSegsRefineMorph,
    "XJSegsDilateMaskExpand": XJSegsDilateMaskExpand,
    "XJMaskDilateCircular": XJMaskDilateCircular,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJMaskRefineKMeans": "Mask Refine (K-Means)",
    "XJMaskRefineMorph": "Mask Refine (Morph)",
    "XJSegsRefineKMeans": "SEGS Refine (K-Means)",
    "XJSegsRefineMorph": "SEGS Refine (Morph)",
    "XJSegsDilateMaskExpand": "Dilate Mask (SEGS Expand)",
    "XJMaskDilateCircular": "Mask Dilate (Circular)",
}
