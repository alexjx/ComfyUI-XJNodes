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

    def test_extract_nested_json_with_unescaped_quotes_is_repaired(self):
        nested = (
            '{"face":"55193591.jpg","prompt":"A girl says "hello" loudly",'
            '"steps":20}'
        )
        payload = {"xj_metadata": nested}

        prompt_out = self.extractor.extract(json.dumps(payload), "xj_metadata.prompt")
        steps_out = self.extractor.extract(json.dumps(payload), "xj_metadata.steps")

        self.assertEqual(prompt_out[0], 'A girl says "hello" loudly')
        self.assertEqual(steps_out[2], 20)

    def test_extract_utf8_and_unicode_escaped_input(self):
        text = "你好 café 😀"
        payload = {"prompt": text}
        for json_string in (
            json.dumps(payload),
            json.dumps(payload, ensure_ascii=False),
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ):
            with self.subTest(json_string=json_string):
                self.assertEqual(self.extractor.extract(json_string, "prompt")[0], text)

    def test_extract_repaired_nested_json_decodes_unicode_escapes(self):
        nested = (
            r'{"\u6807\u9898":"\u4f60\u597d\ud83d\ude00 café says "hello"",'
            r'"steps":20}'
        )
        payload = {"metadata": nested}
        out = self.extractor.extract(json.dumps(payload), "metadata.标题")
        self.assertEqual(out[0], '你好😀 café says "hello"')

    def test_extract_repaired_nested_json_preserves_literal_backslashes(self):
        text = r'你好 café 😀 C:\new\test literal \u4f60 says "hello"'
        nested = json.dumps({"prompt": text}, ensure_ascii=False).replace(
            r'\"hello\"', '"hello"'
        )
        out = self.extractor.extract(json.dumps({"metadata": nested}), "metadata.prompt")
        self.assertEqual(out[0], text)

    def test_extract_repaired_nested_json_invalid_unicode_returns_none_tuple(self):
        for escape in (r"\uZZZZ", r"\u123"):
            with self.subTest(escape=escape):
                nested = '{"prompt":"' + escape + ' says "hello""}'
                out = self.extractor.extract(
                    json.dumps({"metadata": nested}), "metadata.prompt"
                )
                self.assertEqual(out, (None, None, None, None))

    def test_extract_invalid_path_returns_none_tuple(self):
        payload = {"xj_metadata": json.dumps({"prompt": "hello"})}
        out = self.extractor.extract(json.dumps(payload), "xj_metadata.missing")
        self.assertEqual(out, (None, None, None, None))

    def test_extract_unrepairable_nested_json_returns_none_tuple(self):
        payload = {"xj_metadata": '{"prompt": ???}'}
        out = self.extractor.extract(json.dumps(payload), "xj_metadata.prompt")
        self.assertEqual(out, (None, None, None, None))


if __name__ == "__main__":
    unittest.main()
