import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "xj_image_batch", Path(__file__).resolve().parents[1] / "nodes/image/batch.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestImageBatchResize(unittest.TestCase):
    def setUp(self):
        self.node = module.XJImageBatchResize()

    def test_pad_preserves_geometry_and_centers(self):
        for shape, region in [((1, 2, 4, 3), (slice(2, 6), slice(None))),
                              ((1, 4, 2, 3), (slice(None), slice(2, 6)))]:
            with self.subTest(shape=shape):
                result, = self.node.batch(torch.ones(shape), 8, 8, "pad")
                expected = torch.zeros(1, 8, 8, 3)
                expected[:, region[0], region[1], :] = 1
                torch.testing.assert_close(result, expected)

    def test_crop_centers_content_without_stretching(self):
        source = torch.zeros(1, 4, 8, 3)
        source[:, :, 2:6] = 1
        result, = self.node.batch(source, 8, 8, "crop")
        # Independent reference: 2x uniform scaling, then central square.
        expected = torch.nn.functional.interpolate(
            source.movedim(-1, 1), scale_factor=2, mode="bilinear",
            align_corners=False, antialias=True,
        )[:, :, :, 4:12].movedim(1, -1)
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape, (1, 8, 8, 3))
        self.assertGreater(result.mean().item(), 0.9)

    def test_optional_none_gaps_and_batch_order(self):
        first = torch.full((2, 4, 4, 3), 0.25)
        last = torch.full((1, 2, 2, 3), 0.75, dtype=torch.float64)
        result, = self.node.batch(first, 4, 4, "pad", image_2=None, image_8=last)
        self.assertEqual(result.shape, (3, 4, 4, 3))
        torch.testing.assert_close(result[:2], first)
        torch.testing.assert_close(result[2:], torch.full((1, 4, 4, 3), 0.75))
        torch.testing.assert_close(first, torch.full_like(first, 0.25))

    def test_single_input_identity(self):
        source = torch.rand(2, 5, 7, 3)
        for mode in ("pad", "crop"):
            result, = self.node.batch(source, 7, 5, mode)
            torch.testing.assert_close(result, source, rtol=0, atol=0)

    def test_odd_sizes_downscale_and_extreme_ratios(self):
        for h, w in ((3, 17), (17, 3), (1, 100), (100, 1)):
            for mode in ("pad", "crop"):
                result, = self.node.batch(torch.ones(1, h, w, 3), 5, 7, mode)
                self.assertEqual(result.shape, (1, 7, 5, 3))
                self.assertTrue(torch.isfinite(result).all())
                if mode == "crop":
                    torch.testing.assert_close(result, torch.ones_like(result))

    def test_empty_batch(self):
        result, = self.node.batch(torch.empty(0, 2, 3, 3), 5, 7, "pad")
        self.assertEqual(result.shape, (0, 7, 5, 3))

    def test_invalid_inputs(self):
        source = torch.ones(1, 2, 2, 3)
        for args in ((None, 4, 4, "pad"), (source, 0, 4, "pad"),
                     (source, 4, 4, "stretch"), (torch.ones(2, 2, 3), 4, 4, "pad")):
            with self.subTest(args=args[1:]), self.assertRaises(ValueError):
                self.node.batch(*args)
        with self.assertRaisesRegex(ValueError, "channels"):
            self.node.batch(source, 4, 4, "pad", image_2=torch.ones(1, 2, 2, 4))

    def test_registration_and_optional_ports(self):
        self.assertIs(module.NODE_CLASS_MAPPINGS["XJImageBatchResize"], type(self.node))
        inputs = self.node.INPUT_TYPES()
        self.assertIn("image_1", inputs["required"])
        self.assertEqual(set(inputs["optional"]), {f"image_{i}" for i in range(2, 9)})


if __name__ == "__main__":
    unittest.main()
