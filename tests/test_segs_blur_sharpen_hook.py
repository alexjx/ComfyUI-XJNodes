import unittest
import torch
import importlib.util
from pathlib import Path

HOOKS_PATH = Path(__file__).resolve().parents[1] / "nodes" / "segs" / "hooks.py"
spec = importlib.util.spec_from_file_location("xjnodes_segs_hooks", HOOKS_PATH)
hooks = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hooks)


class TestSegsBlurSharpenHook(unittest.TestCase):
    def test_blur_provider_creates_hook(self):
        provider = hooks.XJSegsBlurHookProvider()
        (hook,) = provider.create_hook(radius=1, strength=1.0, feather=0)
        self.assertIsInstance(hook, hooks.BlurSharpenDetailerHook)
        self.assertEqual(hook.mode, "blur")

    def test_sharpen_provider_creates_hook(self):
        provider = hooks.XJSegsSharpenHookProvider()
        (hook,) = provider.create_hook(radius=1, strength=1.0, feather=0)
        self.assertIsInstance(hook, hooks.BlurSharpenDetailerHook)
        self.assertEqual(hook.mode, "sharpen")

    def test_sharpen_provider_defaults_are_visible(self):
        provider = hooks.XJSegsSharpenHookProvider()
        defaults = provider.INPUT_TYPES()["required"]
        radius = defaults["radius"][1]["default"]
        strength = defaults["strength"][1]["default"]

        x = torch.linspace(0.0, 1.0, 128, dtype=torch.float32)
        grad = x.unsqueeze(0).repeat(128, 1)
        texture = 0.03 * torch.sin(torch.linspace(0.0, 60.0, 128, dtype=torch.float32)).unsqueeze(0).repeat(128, 1)
        image = (grad + texture).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

        (hook,) = provider.create_hook(radius=radius, strength=strength, feather=0)
        out = hook.post_decode(image)
        mean_delta = float((out - image).abs().mean())

        self.assertGreater(mean_delta, 0.001)

    def test_blur_without_mask_affects_neighbors(self):
        hook = hooks.BlurSharpenDetailerHook(mode="blur", radius=1, strength=1.0, feather=0)

        pixels = torch.zeros((1, 5, 5, 3), dtype=torch.float32)
        pixels[:, 2, 2, :] = 1.0

        result = hook.post_decode(pixels)

        self.assertGreater(float(result[0, 2, 1, 0]), 0.0)
        self.assertLess(float(result[0, 2, 2, 0]), 1.0)

    def test_blur_with_mask_only_changes_masked_area(self):
        hook = hooks.BlurSharpenDetailerHook(mode="blur", radius=1, strength=1.0, feather=0)

        pixels = torch.zeros((1, 5, 5, 3), dtype=torch.float32)
        pixels[:, 2, 2, :] = 1.0

        mask = torch.zeros((1, 5, 5), dtype=torch.float32)
        mask[:, 2, 2] = 1.0
        hook.post_upscale(pixels, mask)

        result = hook.post_decode(pixels)

        self.assertEqual(float(result[0, 2, 1, 0]), 0.0)
        self.assertLess(float(result[0, 2, 2, 0]), 1.0)

    def test_sharpen_strength_has_visible_separation(self):
        x = torch.linspace(0.0, 1.0, 128, dtype=torch.float32)
        grad = x.unsqueeze(0).repeat(128, 1)
        texture = 0.02 * torch.sin(torch.linspace(0.0, 50.0, 128, dtype=torch.float32)).unsqueeze(0).repeat(128, 1)
        image = (grad + texture).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

        s1 = hooks.apply_blur_or_sharpen(image, mode="sharpen", radius=1, strength=1.0)
        s2 = hooks.apply_blur_or_sharpen(image, mode="sharpen", radius=1, strength=2.0)

        # Strength 2.0 should be measurably stronger than 1.0 on fine texture.
        mean_diff = float((s2 - s1).abs().mean())
        self.assertGreater(mean_diff, 5e-4)


if __name__ == "__main__":
    unittest.main()
