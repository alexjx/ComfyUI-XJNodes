import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from test_load_image_with_metadata import load_module

loaders = load_module()


class TestRandomImageFromDir(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for i in range(4):
            Image.new('RGB', (5, 3), (i * 50, 0, 0)).save(self.directory / f'{i}.png')
        loaders.folder_paths.get_input_directory = lambda: str(self.directory)
        loaders.folder_paths.get_output_directory = lambda: str(self.directory)
        self.node = loaders.XJRandomImageFromDir()
        self.node._rng.seed(42)

    def draw(self, mode='random', recursive=False):
        return self.node.load_image("input", mode=mode, recursive=recursive)["result"]

    def test_random_cycles_and_boundary(self):
        paths = [self.draw()[2] for _ in range(40)]
        for start in range(0, 40, 4):
            self.assertEqual(len(set(paths[start:start + 4])), 4)
        self.assertTrue(all(a != b for a, b in zip(paths, paths[1:])))

    def test_iteration_random_start_and_wrap(self):
        with patch.object(self.node._rng, 'randrange', return_value=2) as rng:
            names = [Path(self.draw('iteration')[2]).name for _ in range(10)]
        self.assertEqual(names, ['2.png', '3.png', '0.png', '1.png'] * 2 + ['2.png', '3.png'])
        rng.assert_called_once_with(4)

    def test_decode_rgb_and_mask(self):
        image, mask, _ = self.draw()
        self.assertEqual(tuple(image.shape), (1, 3, 5, 3))
        self.assertEqual(tuple(mask.shape), (1, 3, 5))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(mask.count_nonzero(), 0)

    def test_alpha_and_first_frame(self):
        for file in self.directory.iterdir():
            file.unlink()
        Image.new('RGBA', (2, 3), (255, 0, 0, 128)).save(self.directory / 'alpha.png')
        image, mask, _ = self.draw()
        self.assertAlmostEqual(mask[0, 0, 0].item(), 127 / 255, places=6)
        self.assertEqual(image[0, 0, 0].tolist(), [1, 0, 0])
        (self.directory / 'alpha.png').unlink()
        Image.new('RGB', (2, 3), 'red').save(
            self.directory / 'animated.gif', save_all=True,
            append_images=[Image.new('RGB', (2, 3), 'blue')])
        self.assertEqual(tuple(self.draw()[0].shape), (1, 3, 2, 3))

    def test_recursive_filtering_and_file_changes(self):
        nested = self.directory / 'nested.png'
        nested.mkdir()
        Image.new('RGB', (2, 2)).save(nested / 'extra.PNG')
        (self.directory / 'notes.txt').write_text('ignored')
        self.assertEqual(len({self.draw()[2] for _ in range(4)}), 4)
        self.assertEqual(len({self.draw(recursive=True)[2] for _ in range(5)}), 5)
        for file in self.directory.glob('[0-9].png'):
            file.unlink()
        paths = [self.draw(recursive=True)[2] for _ in range(3)]
        self.assertEqual(paths, [str(nested / 'extra.PNG')] * 3)

    def test_preview_and_subdirectory(self):
        nested = self.directory / 'nested'
        nested.mkdir()
        Image.new('RGB', (2, 2)).save(nested / 'test.png')
        result = self.node.load_image('output', 'nested')
        self.assertEqual(result['ui']['images'], [
            {'filename': 'test.png', 'subfolder': 'nested', 'type': 'output'}])
        self.assertEqual(result['result'][2], str(nested / 'test.png'))
        with self.assertRaises(ValueError):
            self.node.load_image('input', '..')

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.node.load_image('')
        with self.assertRaises(ValueError):
            self.draw('invalid')
        with self.assertRaises(FileNotFoundError):
            self.node.load_image('input', 'missing')
        for file in self.directory.iterdir():
            file.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'No supported'):
            self.draw()

    def test_failure_does_not_advance(self):
        with patch.object(self.node._rng, 'randrange', return_value=0):
            (self.directory / '0.png').write_bytes(b'broken')
            with self.assertRaises(OSError):
                self.draw('iteration')
            Image.new('RGB', (2, 2)).save(self.directory / '0.png')
            self.assertEqual(Path(self.draw('iteration')[2]).name, '0.png')

    def test_instances_are_independent_and_always_execute(self):
        other = loaders.XJRandomImageFromDir()
        other._rng.seed(42)
        expected = other.load_image("input")["result"][2]
        self.assertEqual(self.draw()[2], expected)
        self.draw()
        self.assertEqual(len(other._remaining), 3)
        self.assertTrue(math.isnan(self.node.IS_CHANGED()))
        self.assertIs(loaders.NODE_CLASS_MAPPINGS['XJRandomImageFromDir'], type(self.node))


if __name__ == '__main__':
    unittest.main()
