import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nodes" / "image" / "loaders.py"


def load_module():
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy.utils = comfy_utils

    folder_paths = types.ModuleType("folder_paths")
    node_helpers = types.ModuleType("node_helpers")
    node_helpers.pillow = lambda function, *args: function(*args)

    stubs = {
        "comfy": comfy,
        "comfy.utils": comfy_utils,
        "folder_paths": folder_paths,
        "node_helpers": node_helpers,
    }
    original_modules = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)

    try:
        spec = importlib.util.spec_from_file_location(
            "xjnodes_image_loaders", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


loaders = load_module()


class TestLoadImageWithMetadata(unittest.TestCase):
    def test_loads_image_directly_in_selected_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "direct.png"
            Image.new("RGB", (4, 3), color="red").save(image_path)
            loaders.folder_paths.get_input_directory = lambda: tmpdir

            result = loaders.XJLoadImageWithMetadata().load_image(
                "input", "", "direct.png"
            )

            self.assertEqual(result[2], "direct.png")
            self.assertEqual((result[3], result[4]), (4, 3))

    def test_rejects_image_from_nested_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested"
            nested_dir.mkdir()
            Image.new("RGB", (4, 3), color="blue").save(nested_dir / "nested.png")
            loaders.folder_paths.get_input_directory = lambda: tmpdir

            validation = loaders.XJLoadImageWithMetadata.VALIDATE_INPUTS(
                "input", "", "nested/nested.png"
            )

            self.assertNotEqual(validation, True)
            with self.assertRaises(ValueError):
                loaders.XJLoadImageWithMetadata().load_image(
                    "input", "", "nested/nested.png"
                )

    def test_loads_direct_image_from_explicitly_selected_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            selected_dir = Path(tmpdir) / "selected"
            selected_dir.mkdir()
            Image.new("RGB", (5, 2), color="green").save(selected_dir / "direct.png")
            loaders.folder_paths.get_input_directory = lambda: tmpdir

            result = loaders.XJLoadImageWithMetadata().load_image(
                "input", "selected", "direct.png"
            )

            self.assertEqual(result[2], "direct.png")
            self.assertEqual((result[3], result[4]), (5, 2))


if __name__ == "__main__":
    unittest.main()
