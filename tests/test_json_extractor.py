import importlib.util
import json
import unittest
from pathlib import Path


JSON_NODE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "text" / "json.py"
spec = importlib.util.spec_from_file_location("xjnodes_text_json", JSON_NODE_PATH)
json_node = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(json_node)


class TestJSONExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = json_node.XJJSONExtractor()

    def test_extract_nested_json_string_success(self):
        payload = {
            "xj_metadata": json.dumps({"prompt": "hello", "steps": 20}),
        }
        out = self.extractor.extract(json.dumps(payload), "xj_metadata.prompt")
        self.assertEqual(out[0], "hello")

    def test_extract_nested_json_trailing_comma_is_tolerated(self):
        nested = '{"prompt":"hello",}'
        payload = {"xj_metadata": nested}
        out = self.extractor.extract(json.dumps(payload), "xj_metadata.prompt")
        self.assertEqual(out[0], "hello")

    def test_extract_invalid_path_returns_none_tuple(self):
        payload = {"xj_metadata": json.dumps({"prompt": "hello"})}
        out = self.extractor.extract(json.dumps(payload), "xj_metadata.missing")
        self.assertEqual(out, (None, None, None, None))


if __name__ == "__main__":
    unittest.main()
