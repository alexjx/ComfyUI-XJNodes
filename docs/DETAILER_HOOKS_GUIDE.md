# Understanding Detailers and Detailer Hooks

A guide to understanding Impact Pack's detailer system and how hooks extend it.

## What is a Detailer?

A **detailer** is an Impact Pack node that:
1. Takes detected segments (SEGS) from object/face detectors
2. Processes each segment individually (upscale, inpaint, enhance)
3. Stitches the processed segments back into the original image

**Common detailer nodes:**
- `FaceDetailer` - Enhances faces
- `DetailerForEach` - Generic segment processing
- `DetailerForEachPipe` - With pipeline support

## The Detailing Pipeline

Here's what happens inside a detailer for each segment:

```
1. Crop segment from original image
   ↓
2. Upscale crop (guide_size)
   ↓
3. Encode to latent
   ↓
4. KSample (inpaint/enhance)
   ↓
5. Decode back to pixels
   ↓
6. Downscale to original crop size
   ↓
7. Paste back into full image
```

## What are Detailer Hooks?

**Detailer hooks** are objects that let you inject custom processing at specific points in the detailing pipeline **without modifying the detailer node itself**.

Think of hooks as "callbacks" or "middleware" that the detailer calls at certain steps.

### Hook Points

A detailer hook can intercept at these points:

| Hook Method | When Called | Purpose |
|-------------|-------------|---------|
| `post_upscale(pixels, mask)` | After upscaling crop, before encoding | Modify upscaled image |
| `post_encode(latent)` | After encoding to latent | Modify latent before sampling |
| `pre_ksample(...)` | Before each sampling step | Change sampling parameters |
| `cycle_latent(latent)` | At start of each cycle | Inject noise or modify latent |
| `pre_decode(latent)` | Before decoding to pixels | Modify final latent |
| `post_decode(pixels)` | After decoding, before downscale | **Modify enhanced pixels** |
| `post_detection(segs)` | After detection, before processing | Filter segments |
| `post_paste(image)` | After pasting segment back | Modify full image |

### Most Common: post_decode

**`post_decode(pixels)`** is the most useful hook point for color/style processing:
- **Input**: Enhanced/inpainted crop (decoded from latent)
- **Output**: Modified crop
- **Use cases**: Color matching, style transfer, detail enhancement

This is where our Color Match Hook operates!

## Hook Workflow Example

### Without Hook
```
Original Crop → Upscale → Encode → Sample → Decode → Enhanced Crop
```

### With Color Match Hook
```
Original Crop → Upscale → [post_upscale: save reference]
                            ↓
                         Encode → Sample → Decode
                            ↓
                    [post_decode: match colors to reference]
                            ↓
                       Color-Matched Crop
```

## Creating a Hook Provider

Hook providers are ComfyUI nodes that create hook instances:

```python
class MyHookProvider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("DETAILER_HOOK",)
    FUNCTION = "create_hook"

    def create_hook(self, param):
        hook = MyDetailerHook(param)
        return (hook,)


class MyDetailerHook:
    def __init__(self, param):
        self.param = param

    def post_decode(self, pixels):
        # Your processing logic here
        return modified_pixels

    # Implement other hook methods (can be no-ops)
    def post_upscale(self, pixels, mask=None):
        return pixels
    # ... etc
```

## Using Hooks

### Connect to Detailer

```
┌────────────────────┐
│  Hook Provider     │ ← Configure parameters
└─────────┬──────────┘
          │ DETAILER_HOOK
┌─────────┴──────────┐
│  FaceDetailer      │ ← Connect to detailer_hook input
│  (has optional     │
│   detailer_hook    │
│   input)           │
└────────────────────┘
```

### Combine Multiple Hooks

Use `DetailerHookCombine` to chain hooks:

```
Hook1 (Color Match) ──┐
                      ├─→ DetailerHookCombine ─→ FaceDetailer
Hook2 (Preview)     ──┘
```

Hooks are executed in order:
1. Hook1.post_upscale → Hook2.post_upscale
2. Hook1.post_decode → Hook2.post_decode
3. etc.

## Built-in Impact Pack Hooks

| Hook | Purpose |
|------|---------|
| `PreviewDetailerHook` | Show processing preview |
| `SEGSOrderedFilterDetailerHook` | Filter segments by order |
| `SEGSRangeFilterDetailerHook` | Filter by size/position |
| `SEGSLabelFilterDetailerHook` | Filter by label |
| `LamaRemoverDetailerHook` | Remove objects with LAMA |
| `SimpleDetailerDenoiseSchedulerHook` | Schedule denoise strength |
| `VariationNoiseDetailerHook` | Add noise variation |
| `BlackPatchRetryHook` | Retry if output is black |
| `SkipEmptyControlImageHook` | Skip empty controlnet images |

## XJNodes Hooks

| Hook | Purpose |
|------|---------|
| `XJSegsColorMatchHookProvider` | Match enhanced colors to original |

## When to Use Hooks

### Use Hooks When:
- You want to modify images/latents during detailing
- You need to inject processing without forking detailer code
- You want reusable processing that works with any detailer
- You need access to intermediate states

### Don't Use Hooks When:
- You want to process the final output (just use regular nodes after detailer)
- You're doing pre-processing (do it before the detector)
- The operation doesn't need access to intermediate states

## Implementation Requirements

A complete detailer hook must implement these methods:

### Core PixelKSampleHook Methods
```python
def set_steps(self, info)                    # Receive (cur_step, total_step)
def post_upscale(self, pixels, mask=None)    # After upscale
def post_encode(self, samples)               # After VAE encode
def pre_decode(self, samples)                # Before VAE decode
def post_decode(self, pixels)                # After VAE decode
def pre_ksample(...)                         # Before sampling (9 params)
def post_crop_region(...)                    # Modify crop region
def touch_scaled_size(self, w, h)            # Modify target size
```

### DetailerHook Additional Methods
```python
def cycle_latent(self, latent)               # At cycle start
def post_detection(self, segs)               # After detection
def post_paste(self, image)                  # After paste
def get_custom_noise(...)                    # Custom noise injection
def get_custom_sampler(self)                 # Custom sampler
def get_skip_sampling(self)                  # Skip sampling flag
def should_retry_patch(self, patch)          # Retry logic
def should_skip_by_cnet_image(self, image)   # Skip by controlnet
```

**Most methods can be no-ops** - only implement what you need!

## Debugging Tips

### Hook Not Running?
- Check it's connected to `detailer_hook` input
- Verify the DETAILER_HOOK type is registered
- Print debug messages in hook methods

### Unexpected Results?
- Add prints to see which hooks are called and when
- Check input/output shapes in your hook methods
- Test with a single segment first

### Performance Issues?
- Profile your hook methods (they run per segment)
- Consider using `post_paste` instead of `post_decode` if you need full image
- Cache expensive computations

## Advanced: Hook State

Hooks can maintain state across calls:

```python
class StatefulHook:
    def __init__(self):
        self.segment_count = 0
        self.saved_reference = None

    def post_upscale(self, pixels, mask=None):
        # Save for later use
        self.saved_reference = pixels.clone()
        return pixels

    def post_decode(self, pixels):
        # Use saved reference
        if self.saved_reference is not None:
            pixels = process(pixels, self.saved_reference)
            self.saved_reference = None  # Clean up
        return pixels
```

**Warning**: State persists across segments in the same detailing pass but not across workflow runs.

## Resources

- **Impact Pack**: https://github.com/ltdrdata/ComfyUI-Impact-Pack
- **Hook Examples**: See Impact Pack's `modules/impact/hooks.py`
- **XJNodes Color Match Hook**: See `SEGS_COLOR_MATCH_HOOK.md`

## Summary

- **Detailers** = Process image segments individually
- **Hooks** = Inject custom logic into the detailing pipeline
- **Hook Providers** = ComfyUI nodes that create hooks
- **Most useful hook point** = `post_decode` (after enhancement, before paste)
- **Hooks are composable** = Use `DetailerHookCombine` to chain them

Hooks are powerful because they let you extend detailers without modifying their code, and they work with **any** detailer node that supports the `detailer_hook` input.
