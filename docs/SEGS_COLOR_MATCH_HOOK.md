# SEGS Color Match Hook

A detailer hook for Impact Pack that applies color matching to enhanced/inpainted SEGS crops, preventing color shifts during detailing operations.

## Overview

When using Impact Pack's detailer nodes (like `FaceDetailer`, `DetailerForEachPipe`), the enhanced/inpainted regions often have color shifts compared to the original image. This hook automatically matches the colors of the enhanced crops to the original crops, ensuring consistent color throughout the image.

## How It Works

The hook integrates into Impact Pack's detailing pipeline:

1. **`post_upscale`**: Saves the original crop (after upscaling, before inpainting) as reference
2. Detailer performs inpainting/enhancement
3. **`post_decode`**: Applies color matching to the enhanced image using the saved reference
4. Result is stitched back into the full image

## Node: SEGS Color Match Hook

**Location**: `XJNodes/segs/hooks`

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| method | STRING | mkl | Color matching method |
| strength | FLOAT | 1.0 | Blend strength (0.0 = no change, 1.0 = full match) |

### Methods

| Method | Description | Dependencies |
|--------|-------------|--------------|
| **adain** | Adaptive Instance Normalization - Fast, matches mean/std statistics | None |
| **mkl** | Monge-Kantorovitch Linear transfer | color-matcher |
| **hm** | Histogram Matching | color-matcher |
| **reinhard** | Reinhard color transfer | color-matcher |
| **mvgd** | Mean Variance and Gradient Deviation | color-matcher |
| **hm-mvgd-hm** | Combined histogram-mvgd-histogram | color-matcher |
| **hm-mkl-hm** | Combined histogram-mkl-histogram | color-matcher |

**Default**: `mkl` (Monge-Kantorovitch Linear) provides excellent color matching quality. Use `adain` if you prefer faster processing with no external dependencies.

### Output

| Output | Type | Description |
|--------|------|-------------|
| hook | DETAILER_HOOK | Hook instance to connect to Impact Pack detailer nodes |

## Installation

### No Dependencies (adain only)
The `adain` method works out of the box with no additional dependencies.

### Color-Matcher Methods
To use `mkl`, `hm`, `reinhard`, `mvgd`, and combined methods:

```bash
pip install color-matcher
```

## Usage Example

### Basic Usage with FaceDetailer

```
┌─────────────────────────┐
│  Load Image             │
└───────────┬─────────────┘
            │
┌───────────┴─────────────┐
│  BBox Detector          │
│  (detect faces)         │
└───────────┬─────────────┘
            │ SEGS
┌───────────┴─────────────┐
│  SEGS Color Match Hook  │
│  - method: mkl          │
│  - strength: 1.0        │
└───────────┬─────────────┘
            │ hook
┌───────────┴─────────────┐
│  FaceDetailer           │
│  (connect hook to       │
│   detailer_hook input)  │
└───────────┬─────────────┘
            │
┌───────────┴─────────────┐
│  Save Image             │
└─────────────────────────┘
```

### With Strength Control

For subtle color correction, use `strength < 1.0`:

```python
strength: 0.5  # 50% color match, 50% original enhanced colors
strength: 0.8  # 80% color match, 20% original enhanced colors
strength: 1.0  # 100% color match (default)
```

### Combining with Other Hooks

The color match hook can be combined with other Impact Pack hooks using `DetailerHookCombine`:

```
┌────────────────────────────┐
│  SEGS Color Match Hook     │
└─────────────┬──────────────┘
              │ hook1
┌─────────────┴──────────────┐
│  PreviewDetailerHook       │
└─────────────┬──────────────┘
              │ hook2
┌─────────────┴──────────────┐
│  DetailerHookCombine       │
│  (hook1 + hook2)           │
└─────────────┬──────────────┘
              │ combined_hook
┌─────────────┴──────────────┐
│  FaceDetailer              │
└────────────────────────────┘
```

## Technical Details

### Why Not Wavelet?

Wavelet color transfer is not supported because:
- The enhanced image has different content than the original (due to inpainting)
- Wavelet decomposition depends on spatial structure alignment
- Misaligned structures cause artifacts

We use statistical methods instead (adain, histogram matching) which only care about color distributions, not spatial structure.

### Size Handling

The hook automatically handles size mismatches:
- Reference is saved at upscaled size
- Enhanced image may have different dimensions
- Reference is automatically resized to match enhanced image before color matching

### Performance

- **adain**: Very fast (~5-10ms for 512x512 image)
- **color-matcher methods**: Slower (~50-200ms), but higher quality for some cases

## Troubleshooting

### "ModuleNotFoundError: color-matcher"
Install the color-matcher library: `pip install color-matcher`

### Colors still don't match
- Try different methods (adain vs hm vs mkl)
- Adjust strength parameter
- Check if the original image has extreme colors

### Hook not working
- Ensure the hook is connected to the `detailer_hook` input of Impact Pack detailer nodes
- Verify you're using a compatible Impact Pack version
- Check that DETAILER_HOOK type is recognized

## Compatibility

- **Requires**: ComfyUI-Impact-Pack (for DETAILER_HOOK type)
- **Optional**: color-matcher library (for non-adain methods)
- **Tested with**: Impact Pack v6.0+

## Implementation Notes

The hook implements all required Impact Pack DetailerHook methods:
- Core hooks: `post_upscale`, `post_decode` (color matching logic)
- Pass-through hooks: `post_encode`, `pre_decode`, `pre_ksample`, `cycle_latent`, etc.
- Compatibility hooks: `post_detection`, `post_paste`, `get_custom_noise`, etc.

## Examples

### Before/After

**Without Hook:**
- Original image: Natural skin tones
- Enhanced face: Color shifted towards model's training data (often more saturated/different hue)

**With Hook:**
- Original image: Natural skin tones
- Enhanced face: Maintains original color tone while preserving enhanced details

### Use Cases

1. **Face Enhancement**: Prevent color shifts when enhancing faces
2. **Inpainting**: Maintain color consistency when inpainting regions
3. **Style Preservation**: Keep original color style during upscaling
4. **Iterative Detailing**: Maintain color across multiple detailing passes

## Contributing

Found a bug or want to add more color matching methods? PRs welcome!

## License

Same as ComfyUI-XJNodes (MIT)
