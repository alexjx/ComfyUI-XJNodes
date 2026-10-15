import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "xjnodes_text_memory", Path(__file__).resolve().parents[1] / "nodes/text/loaders.py"
)
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"folder_paths": types.ModuleType("folder_paths")}):
    spec.loader.exec_module(module)
Node = module.XJRandomTextFromList


class TestRandomTextMemory(unittest.TestCase):
    def setUp(self):
        Node._buckets.clear()
        self.text = "\n".join(f"- item{i}" for i in range(60))

    def draw(self, seed=0, **kwargs):
        options = dict(multiline_text=self.text, type="list", choice=1,
                       seed=seed, index_list="1-60", memory=True, unique_id="7")
        options.update(kwargs)
        # Deliberately construct a new instance for every execution.
        return Node().make_list(**options)[0]

    def bucket(self):
        return Node._buckets[("", "7", "list")]

    def test_cross_instance_memory_and_strict_quarter_boundary(self):
        outputs = [self.draw(seed) for seed in range(45)]
        self.assertEqual(len(set(outputs)), 45)
        self.assertEqual(len(self.bucket()["remaining"]), 15)
        outputs.append(self.draw(45))
        self.assertEqual(len(set(outputs)), 46)
        self.assertEqual(len(self.bucket()["remaining"]), 14)
        last = outputs[-1]
        self.assertNotEqual(self.draw(46), last)
        self.assertEqual(len(self.bucket()["remaining"]), 59)

    def test_timestamp_refresh_and_expiration(self):
        with patch.object(module.time, "monotonic", return_value=0):
            self.draw(cache_time=600)
            stale = self.bucket()
        with patch.object(module.time, "monotonic", return_value=600):
            self.draw(unique_id="8")
            self.assertIs(self.bucket(), stale)
        with patch.object(module.time, "monotonic", return_value=601):
            self.draw(unique_id="8")
            self.assertNotIn(("", "7", "list"), Node._buckets)
            self.assertEqual(Node._buckets[("", "8", "list")]["timestamp"], 601)
        with patch.object(module.time, "monotonic", return_value=602):
            self.draw()
            self.assertEqual(len(self.bucket()["remaining"]), 59)

    def test_default_timeout_and_widget(self):
        widget = Node.INPUT_TYPES()["optional"]["cache_time"]
        self.assertEqual(widget[0], "INT")
        self.assertEqual(widget[1]["default"], 7200)
        self.assertEqual(widget[1]["min"], 1)
        with patch.object(module.time, "monotonic", return_value=0):
            self.draw()
            original = self.bucket()
        with patch.object(module.time, "monotonic", return_value=7200):
            self.draw(unique_id="8", cache_time=1)
            self.assertIs(self.bucket(), original)
        with patch.object(module.time, "monotonic", return_value=7201):
            self.draw()
            self.assertIsNot(self.bucket(), original)

    def test_custom_timeout_refreshes_and_is_isolated(self):
        with patch.object(module.time, "monotonic", return_value=0):
            self.draw(cache_time=10)
            self.draw(unique_id="8", cache_time=100)
        with patch.object(module.time, "monotonic", return_value=10):
            self.draw(cache_time=10)
            self.assertEqual(len(self.bucket()["remaining"]), 58)
        with patch.object(module.time, "monotonic", return_value=21):
            self.draw(unique_id="9", cache_time=1)
            self.assertNotIn(("", "7", "list"), Node._buckets)
            self.assertIn(("", "8", "list"), Node._buckets)

    def test_changed_timeout_applies_on_next_draw(self):
        for initial, updated, resets in ((100, 10, True), (10, 100, False)):
            with self.subTest(initial=initial, updated=updated):
                Node._buckets.clear()
                with patch.object(module.time, "monotonic", return_value=0):
                    self.draw(cache_time=initial)
                    original = self.bucket()
                with patch.object(module.time, "monotonic", return_value=20):
                    self.draw(cache_time=updated)
                    self.assertEqual(self.bucket() is not original, resets)
                    self.assertEqual(self.bucket()["ttl"], updated)

    def test_recent_access_extends_ttl(self):
        for now in (0, 500, 1000):
            with patch.object(module.time, "monotonic", return_value=now):
                self.draw()
        self.assertEqual(len(self.bucket()["remaining"]), 57)
        self.assertEqual(self.bucket()["timestamp"], 1000)

    def test_nodes_and_workflows_are_isolated(self):
        first = self.draw(seed=42)
        self.assertEqual(self.draw(seed=42, unique_id="8"), first)
        self.assertEqual(self.draw(seed=42, extra_pnginfo={"workflow": {"id": "other"}}), first)
        self.assertEqual(self.draw(seed=42, memory_id="api-workflow"), first)
        self.assertEqual(len(Node._buckets), 4)
        self.assertNotEqual(self.draw(seed=42), first)

    def test_candidate_change_resets_in_place(self):
        self.draw()
        self.assertIn(self.draw(index_list="1-4"), [f"item{i}" for i in range(4)])
        self.assertEqual(len(self.bucket()["remaining"]), 3)
        self.assertEqual(len(Node._buckets), 1)
        self.draw(multiline_text=self.text.replace("item0", "changed"), index_list="1-4")
        self.assertIn("changed", self.bucket()["candidates"])
        self.assertEqual(len(self.bucket()["remaining"]), 3)

    def test_duplicate_indices_and_text_do_not_bypass_memory(self):
        outputs = [self.draw(seed, multiline_text="- A\n- A\n- B\n- C",
                             index_list="1, 1-4") for seed in range(2)]
        self.assertEqual(len(set(outputs)), 2)
        self.assertEqual(self.bucket()["candidates"], ("A", "B", "C"))
        self.assertEqual(len(self.bucket()["remaining"]), 1)
        self.assertNotEqual(self.draw(2, multiline_text="- A\n- A\n- B\n- C",
                                     index_list="1, 1-4"), outputs[-1])
        self.assertEqual(len(self.bucket()["remaining"]), 0)
        self.draw(3, multiline_text="- A\n- A\n- B\n- C", index_list="1, 1-4")
        self.assertEqual(len(self.bucket()["remaining"]), 2)

    def test_single_and_two_candidates(self):
        self.assertEqual([self.draw(seed, index_list="1") for seed in range(5)], ["item0"] * 5)
        outputs = [self.draw(seed, index_list="1-2") for seed in range(20)]
        self.assertTrue(all(a != b for a, b in zip(outputs, outputs[1:])))

    def test_memory_off_preserves_seeded_and_weighted_behavior(self):
        for seed in range(50):
            expected = random.Random(seed).choice([0, 0, 1, 2])
            self.assertEqual(self.draw(seed, memory=False, index_list="1, 1-3"), f"item{expected}")
            expected = random.Random(seed).randint(0, 59)
            self.assertEqual(self.draw(seed, memory=False, type="random"), f"item{expected}")
        self.assertFalse(Node._buckets)

    def test_fixed_empty_invalid_and_random_mode(self):
        self.assertEqual(self.draw(type="fixed", choice=61), "item0")
        self.assertEqual(self.draw(multiline_text=""), [""])
        with self.assertRaisesRegex(Exception, "No valid indices"):
            self.draw(index_list="99")
        self.assertFalse(Node._buckets)
        outputs = [self.draw(seed, type="random") for seed in range(46)]
        self.assertEqual(len(set(outputs)), 46)

    def test_same_history_is_reproducible_without_global_rng_changes(self):
        state = random.getstate()
        first = [self.draw(seed) for seed in range(100)]
        Node._buckets.clear()
        second = [self.draw(seed) for seed in range(100)]
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), state)


if __name__ == "__main__":
    unittest.main()
