import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


spec = importlib.util.spec_from_file_location(
    "xj_pair_compare", Path(__file__).resolve().parents[1] / "nodes/image/pair_compare.py"
)
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(PreviewImage=object)}):
    spec.loader.exec_module(module)


class TestImagePairCompare(unittest.TestCase):
    def setUp(self):
        self.node = module.XJImagePairCompare()
        self.node.save_images = Mock(side_effect=lambda images, *args: {
            "ui": {"images": [{"filename": f"preview_{len(images)}.png"}]}
        })

    def test_outputs_original_a_and_keeps_both_previews(self):
        images_a = np.zeros((2, 4, 4, 3))
        images_b = np.ones((2, 4, 4, 3))
        prompt = {"example": {}}
        metadata = {"workflow": {}}

        result = self.node.compare_pairs(images_a, images_b, "compare", prompt, metadata)

        self.assertIs(result["result"][0], images_a)
        self.assertEqual(len(result["result"]), 1)
        self.assertEqual(result["ui"], {
            "a_images": [{"filename": "preview_2.png"}],
            "b_images": [{"filename": "preview_2.png"}],
        })
        self.assertEqual(self.node.save_images.call_count, 2)
        for call, source in zip(self.node.save_images.call_args_list, (images_a, images_b)):
            self.assertIs(call.args[0], source)
            self.assertEqual(call.args[1:], ("compare", prompt, metadata))

    def test_b_only_is_forwarded(self):
        images = np.ones((1, 4, 4, 3))
        result = self.node.compare_pairs(images_b=images)
        self.assertIs(result["result"][0], images)
        self.assertEqual(result["ui"]["a_images"], [])
        self.node.save_images.assert_called_once()

    def test_empty_a_is_preserved(self):
        images_a = np.empty((0, 4, 4, 3))
        result = self.node.compare_pairs(images_a, np.ones((1, 4, 4, 3)))
        self.assertIs(result["result"][0], images_a)
        self.assertEqual(result["ui"]["a_images"], [])
        self.node.save_images.assert_called_once()

    def test_disconnected_inputs_have_a_completion_result(self):
        result = self.node.compare_pairs()
        self.assertEqual(result, {
            "ui": {"a_images": [], "b_images": []}, "result": (None,),
        })
        self.node.save_images.assert_not_called()

    def test_mismatched_pairs_still_fail_before_saving(self):
        with self.assertRaisesRegex(ValueError, "Image count mismatch"):
            self.node.compare_pairs(np.zeros((2, 4, 4, 3)), np.zeros((1, 4, 4, 3)))
        self.node.save_images.assert_not_called()


if __name__ == "__main__":
    unittest.main()
