import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "__init__.py"


def load_module():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(ROOT)
    folder_paths.get_output_directory = lambda: str(ROOT)

    class _Routes:
        def post(self, _path):
            def decorator(func):
                return func

            return decorator

        def get(self, _path):
            def decorator(func):
                return func

            return decorator

    prompt_server = types.SimpleNamespace(routes=_Routes())
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=prompt_server)

    nodes_module = types.ModuleType("xjnodes_nodes")
    nodes_module.NODE_CLASS_MAPPINGS = {}
    nodes_module.NODE_DISPLAY_NAME_MAPPINGS = {}

    package_name = "xjnodes_test_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]

    sys.modules.setdefault("folder_paths", folder_paths)
    sys.modules.setdefault("server", server)
    sys.modules.setdefault(package_name, package)
    sys.modules.setdefault(f"{package_name}.nodes", nodes_module)

    spec = importlib.util.spec_from_file_location(
        package_name,
        MODULE_PATH,
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


xjnodes = load_module()


class TestSaveRatingMetadata(unittest.TestCase):
    def test_write_regular_png_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.png"
            Image.new("RGB", (4, 4), color="red").save(image_path, "PNG")

            success, error = xjnodes._write_rating_metadata(str(image_path), 4, "note")

            self.assertTrue(success, error)
            self.assertFalse(image_path.with_suffix(".png.tmp").exists())

            with Image.open(image_path) as image:
                self.assertEqual(image.text.get(xjnodes.RATING_KEY), "4")
                self.assertEqual(image.text.get(xjnodes.COMMENT_KEY), "note")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unsupported on this platform")
    def test_write_symlink_preserves_link_and_updates_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            target_path = tmpdir / "target.png"
            link_path = tmpdir / "link.png"
            Image.new("RGB", (4, 4), color="blue").save(target_path, "PNG")
            os.symlink(target_path, link_path)

            success, error = xjnodes._write_rating_metadata(str(link_path), 5, "linked")

            self.assertTrue(success, error)
            self.assertTrue(link_path.is_symlink())
            self.assertEqual(os.path.realpath(link_path), str(target_path))
            self.assertFalse((tmpdir / "target.png.tmp").exists())

            with Image.open(target_path) as image:
                self.assertEqual(image.text.get(xjnodes.RATING_KEY), "5")
                self.assertEqual(image.text.get(xjnodes.COMMENT_KEY), "linked")


if __name__ == "__main__":
    unittest.main()
