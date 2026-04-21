import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nodes" / "image" / "savers.py"


def load_module():
    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: "cpu"
    model_management.get_torch_device_name = lambda device=None: "cpu"
    comfy.model_management = model_management

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(ROOT)

    def get_save_image_path(filename_prefix, output_folder, width, height):
        return output_folder, filename_prefix, 1, "", filename_prefix

    folder_paths.get_save_image_path = get_save_image_path

    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.model_management", model_management)
    sys.modules.setdefault("folder_paths", folder_paths)

    spec = importlib.util.spec_from_file_location("xjnodes_image_savers", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeImage:
    def __init__(self, array):
        self._array = array
        self.shape = array.shape

    def cpu(self):
        return self

    def numpy(self):
        return self._array


savers = load_module()


class TestSaveImageMetadata(unittest.TestCase):
    def test_normalize_metadata_repairs_invalid_json(self):
        raw = '{"face":"55193591.jpg","prompt":"A girl says "hello" loudly","steps":20}'

        normalized = savers.normalize_metadata(raw)

        self.assertEqual(
            normalized,
            json.dumps(
                {
                    "face": "55193591.jpg",
                    "prompt": 'A girl says "hello" loudly',
                    "steps": 20,
                },
                ensure_ascii=False,
            ),
        )

    def test_save_images_skips_unrepairable_metadata(self):
        image = FakeImage(np.zeros((4, 4, 3), dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            saver = savers.XJSaveImageWithMetadata()
            saver.output_dir = tmpdir

            output = saver.save_images(
                images=[image],
                output_path="",
                filename_prefix="sample",
                extension="png",
                metadata='{"prompt": ???}',
                overwrite_existing=True,
                embed_workflow=False,
            )

            paths = output["result"][0]
            self.assertEqual(len(paths), 1)

            with Image.open(paths[0]) as saved:
                self.assertNotIn("xj_metadata", saved.text)


if __name__ == "__main__":
    unittest.main()
