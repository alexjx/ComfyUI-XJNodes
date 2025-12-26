# SEGS Architecture Design

## Overview

The SEGS (Segmentation) system provides modular, composable nodes for working with image segments. Unlike monolithic "detailer" approaches, this architecture decomposes segmentation workflows into discrete steps, allowing maximum flexibility with standard ComfyUI nodes.

## Core Principle

**Decomposition over Integration**: Break SEGS processing into atomic operations that expose standard ComfyUI types (IMAGE, MASK) between steps, rather than keeping data encapsulated in SEGS format throughout the pipeline.

## Data Structure

### SEG Definition

```python
SEG = namedtuple("SEG", [
    'cropped_image',      # Tensor: cropped region from original image
    'cropped_mask',       # Tensor: segment mask (0.0-1.0)
    'confidence',         # float: detection confidence
    'crop_region',        # tuple: (x1, y1, x2, y2) crop coordinates
    'bbox',               # tuple: (x1, y1, x2, y2) detected bounds
    'label',              # str: segment class/label
    'control_net_wrapper' # optional: ControlNet data
])
```

### SEGS Format

```python
SEGS = (shape, segments)
# shape: (height, width) - original image dimensions
# segments: [SEG, SEG, ...] - list of segment elements
```

**Key Insight**: `crop_region` defines where the crop was taken (may include padding/context), while `bbox` defines the actual detected object bounds within that crop.

## Design Principles

### 1. Single Responsibility
Each node does one thing well. Filtering, extraction, and stitching are separate concerns.

### 2. Standard Types Between Steps
Export to standard ComfyUI types (IMAGE, MASK) as early as possible. User workflows operate on standard types, not SEGS.

### 3. Index-Based Access
Process one segment at a time with explicit indexing. Enables loop-based workflows without hidden batching complexity.

### 4. No Hidden State
All operations are pure functions. Node behavior depends only on inputs, never on external state.

### 5. Future-Proof Boundaries
The gap between Extractor and Stitcher is where user creativity lives. Never assume what happens there.

## Node Categories

### Input Nodes
Generate or load SEGS data.
- Detectors (future: defer to existing detector nodes)
- Converters (masks → SEGS, etc.)

### Query Nodes
Inspect SEGS without modification.
- Count: return number of segments
- Metadata extraction

### Selection Nodes
Choose subset of SEGS.
- Filter: by size, confidence, position, order
- Pick: select single SEG by index
- Label-based selection (requires label input)

### Decomposition Nodes
Break SEGS into standard ComfyUI types.
- Extractor: SEG + optional IMAGE → IMAGE, MASK, metadata
  - Without image input: returns `seg.cropped_image` (original detector crop)
  - With image input: crops from provided image using `crop_region` (enables iterative workflows)

### Composition Nodes
Reconstruct images from processed segments.
- Stitcher: IMAGE + MASK + metadata → final image

## Data Flow Patterns

### Pattern 1: Single Segment Processing
```
Detector → SEGS Count → SEGS Pick[i] → Extractor → [User Workflow] → Stitcher
```

### Pattern 2: Loop-Based Processing (Iterative)
```
current_image = original_image
For i in range(SEGS Count):
    SEGS Pick[i] → Extractor(current_image, seg) → [User Workflow] → Stitcher(current_image) → current_image
```
**Critical**: Extractor must receive `current_image` to handle overlapping segments correctly. Each iteration sees improvements from previous iterations.

### Pattern 3: Filter-Then-Process
```
Detector → SEGS Filter → SEGS Pick[i] → Extractor → [User Workflow] → Stitcher
```

### Pattern 4: Multi-Stage Selection
```
Detector → Filter by Size → Filter by Confidence → Pick Best → Process
```

## Extension Points

### Adding New Filters
Filters operate on `[SEG]` list and return filtered `[SEG]` list. All filters:
- Accept SEGS input
- Return SEGS output with same shape, different segment list
- Are composable (output of one is input to another)

### Adding New Extractors
Extractors take single SEG and expose its components. Consider:
- What standard types does it output?
- What metadata is needed for reconstruction?
- Can it work with index-based access?

### Adding New Stitchers
Stitchers combine processed segments back to images. Consider:
- Blending strategy (feathering, alpha, etc.)
- Coordinate system handling
- Multi-segment overlap handling
- Auto-resize handling (see Resize Invariant below)

### Adding New Detectors
Detectors create SEGS from images. Must:
- Return SEGS format: `((h, w), [SEG, ...])`
- Populate all SEG fields correctly
- Set crop_region appropriate to use case

## Compatibility Guidelines

### With Impact Pack
- Data structures are compatible (same field names, same tuple structure)
- SEGS from Impact Pack detectors can be used with XJNodes processors
- XJNodes SEGS can be used with Impact Pack nodes

### Internal Compatibility
- All XJNodes SEGS nodes accept/return consistent SEGS format
- No version-specific behavior
- Backward compatibility NOT required (see project CLAUDE.md)

## Coordinate Systems

### crop_region vs bbox
- `crop_region`: Coordinates in **original image space** where crop was extracted
- `bbox`: Coordinates in **cropped image space** where object is detected
- `crop_region` may be larger than `bbox` to include context

### Stitcher Reconstruction
Uses `crop_region` to place processed segment back into original image coordinates.

### Resize Invariant
**The Detailer Enhancement Problem**: In typical detail enhancement workflows, users upscale the extracted segment (e.g., 512x512 → 2048x2048), process it at high resolution, then need to stitch it back.

**Stitcher Auto-Resize**: The stitcher automatically handles size mismatches:
1. Detects if `processed_image` dimensions differ from `crop_region` dimensions
2. Validates aspect ratio hasn't changed (1% tolerance)
3. Throws exception if aspect ratio mismatch detected
4. Auto-resizes both image and mask to match `crop_region` dimensions for placement

**Note**: Validation uses `crop_region` dimensions, not `seg.cropped_image.shape`, because Extractor crops based on `crop_region` in iterative workflows.

**Design Rationale**:
- User controls upscale method (algorithm vs model) - modularity preserved
- Stitcher owns coordinate alignment - it knows the expected dimensions from SEG
- Aspect ratio validation catches user errors (incorrect crops/resizes) early
- No manual size tracking required - reduces workflow complexity

**Assumption**: Aspect ratio is invariant. If the user changes aspect ratio in their workflow, it indicates an error (cropping, incorrect resize) and should fail loudly.

### Iterative Processing for Overlapping Segments

**The Overlapping Segments Problem**: When multiple segments overlap, later segments should see improvements from earlier segments, not the original image.

**Wrong Approach:**
```
seg[0]: Extract seg.cropped_image (from original) → Process → Stitch → v1
seg[1]: Extract seg.cropped_image (from original) → Process → Stitch → v2
        ^^^^^^^^^^^^^^^^ Misses improvements from seg[0]!
```

**Correct Approach:**
```
current = original_image

seg[0]: Extract crop from current → Process → Stitch into current → current updated
seg[1]: Extract crop from current → Process → Stitch into current → current updated
        ^^^^^^^^^^^^^^^ Sees improvements from seg[0]
```

**Implementation**: Extractor accepts optional `image` input. When provided, it crops from that image using `crop_region` instead of returning `seg.cropped_image`. This enables:
- Iterative refinement across segments
- Overlapping regions benefit from previous enhancements
- Each pass builds on previous passes

**Key Insight**: `seg.cropped_image` is just the detector's initial snapshot. The contract is `crop_region` coordinates, not the pixels.

## Quality Principles

### Mask Quality
- Masks should be feathered at boundaries for seamless stitching
- Feathering happens at stitch time, not during extraction
- Preserve original mask data through pipeline

### Image Quality
- Maintain original image color space and precision
- No unnecessary conversions or quality loss
- Respect user's image processing choices in between extraction and stitching

## Non-Goals

### Not Building
- Detectors (use existing detector nodes)
- Inpainting models (use standard ComfyUI nodes)
- Upscalers (use standard ComfyUI nodes)
- Prompt processing (use standard ComfyUI nodes)

### Not Enforcing
- Specific workflows between Extractor and Stitcher
- Particular image enhancement methods
- Standardized processing pipelines

## Future Considerations

### Potential Additions
- Batch processing variants (process all SEGs at once)
- Advanced stitching modes (beyond simple feathering)
- SEGS visualization/debugging nodes
- Multi-image SEGS (tracking segments across frames)

### Principles for Future Nodes
1. Does it operate on SEGS format? → Goes in `nodes/segs/`
2. Does it convert to/from standard types? → Decomposition or Composition
3. Does it modify SEGS list? → Selection
4. Does it inspect SEGS? → Query
5. Does it create SEGS? → Input

### Integration Points
- API endpoints for SEGS inspection (similar to existing `/xjnodes/list_images`)
- Web UI widgets for SEGS visualization
- Workflow templates for common SEGS patterns

## Summary

This architecture prioritizes **flexibility and composability** over convenience. By exposing standard ComfyUI types between processing steps, we enable unlimited user creativity without being constrained by pre-built "detailer" logic. The system is designed to grow organically as new use cases emerge, guided by these core principles.
