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
                "n_clusters": ("INT", {"default": 3, "min": 2, "max": 10, "step": 1}),
                "grow_pixels": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1}),
                "similarity_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
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
                "grow_pixels": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1}),
                "smooth_pixels": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
                "edge_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_edge_detection": ("BOOLEAN", {"default": True}),
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
                "n_clusters": ("INT", {"default": 3, "min": 2, "max": 10, "step": 1}),
                "grow_pixels": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1}),
                "similarity_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
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

            # Ensure mask has batch dimension (1, H, W) for SEGS compatibility
            if len(refined_mask.shape) == 2:
                refined_mask = refined_mask.unsqueeze(0)

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
                "grow_pixels": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1}),
                "smooth_pixels": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
                "edge_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_edge_detection": ("BOOLEAN", {"default": True}),
                "color_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "If > 0, stop growth when pixel color differs from mask average by more than this threshold (0-1). Use instead of edge detection for similar colors."}),
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

            # Ensure mask has batch dimension (1, H, W) for SEGS compatibility
            if len(refined_mask.shape) == 2:
                refined_mask = refined_mask.unsqueeze(0)

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


NODE_CLASS_MAPPINGS = {
    "XJMaskRefineKMeans": XJMaskRefineKMeans,
    "XJMaskRefineMorph": XJMaskRefineMorph,
    "XJSegsRefineKMeans": XJSegsRefineKMeans,
    "XJSegsRefineMorph": XJSegsRefineMorph,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJMaskRefineKMeans": "Mask Refine (K-Means)",
    "XJMaskRefineMorph": "Mask Refine (Morph)",
    "XJSegsRefineKMeans": "SEGS Refine (K-Means)",
    "XJSegsRefineMorph": "SEGS Refine (Morph)",
}
