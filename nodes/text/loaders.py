import os
import random
import string
import hashlib
import folder_paths
from collections import deque


class XJRandomTextFromList:
    # Class-level memory to track recently selected items per unique list
    # Key: hash of list content, Value: deque of recently selected indices
    _selection_memories = {}  # Dict of deques, one per unique list

    @staticmethod
    def parse_index_list(index_string):
        """
        Parse a comma-separated string with range support.
        Examples:
            "3, 5, 7-10" -> [3, 5, 7, 8, 9, 10]
            "1, 3, 5-8, 10" -> [1, 3, 5, 6, 7, 8, 10]
            "10-7" -> [10, 9, 8, 7] (reverse range)

        Returns:
            List of integers (1-indexed as displayed to users)
        """
        indices = []
        parts = index_string.split(',')

        for part in parts:
            part = part.strip()
            if '-' in part:
                # Handle range
                range_parts = part.split('-')
                if len(range_parts) == 2:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    if start <= end:
                        indices.extend(range(start, end + 1))
                    else:
                        # Reverse range
                        indices.extend(range(start, end - 1, -1))
            else:
                # Single number
                indices.append(int(part))

        return indices

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multiline_text": ("XJ_NUMBERED_LIST", {}),
                "type": (("fixed", "random", "list"),),
                "choice": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "step": 1,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "index_list": ("STRING", {"default": "1, 2, 3"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "make_list"
    CATEGORY = "XJNode/text"

    def make_list(self, multiline_text, type, choice, seed, index_list="1, 2, 3"):
        # Split the multiline text into a list of strings
        text_list = multiline_text.splitlines()
        # Remove empty lines
        text_list = [text.strip().lstrip("- ") for text in text_list]
        text_list = [
            text for text in text_list if text and not text.startswith("#")
        ]  # skip comments
        if not text_list:
            return ([""],)

        if type == "fixed":
            # If type is fixed, return the first 'choice' number of strings
            if choice >= len(text_list) + 1:
                raise Exception(f"Choice {choice} exceeded max length {len(text_list)}")
            selected_text = text_list[choice - 1]
        elif type == "list":
            # Parse the index list
            try:
                valid_indices = self.parse_index_list(index_list)
            except (ValueError, IndexError) as e:
                raise Exception(f"Invalid index_list format: {index_list}. Error: {e}")

            # Filter to only valid indices (within bounds)
            valid_indices = [i for i in valid_indices if 1 <= i <= len(text_list)]

            if not valid_indices:
                raise Exception(f"No valid indices in range for list with {len(text_list)} items")

            # Create a unique key for this list based on its content and valid indices
            list_key = hashlib.md5(
                ('|'.join(text_list) + '||' + index_list).encode('utf-8')
            ).hexdigest()

            # Get or create memory deque for this specific list
            if list_key not in self._selection_memories:
                self._selection_memories[list_key] = deque(maxlen=100)
            memory = self._selection_memories[list_key]

            # Create a Random object with the seed for better randomness
            rng = random.Random(seed)

            # Convert to 0-indexed for accessing text_list
            available_indices = [i - 1 for i in valid_indices]

            # Apply anti-repetition logic (same as random mode)
            memory_size = max(1, len(available_indices) // 2 - 1)
            max_retries = min(len(available_indices), 50)
            selected_index = None

            for _ in range(max_retries):
                candidate_index = rng.choice(available_indices)
                if candidate_index not in list(memory)[-memory_size:]:
                    selected_index = candidate_index
                    break

            # If all retries failed (unlikely), just use a random selection
            if selected_index is None:
                selected_index = rng.choice(available_indices)

            # Add selected index to this list's memory
            memory.append(selected_index)

            selected_text = text_list[selected_index]
        else:  # random
            # Create a unique key for this list based on its content
            list_key = hashlib.md5('|'.join(text_list).encode('utf-8')).hexdigest()

            # Get or create memory deque for this specific list
            if list_key not in self._selection_memories:
                self._selection_memories[list_key] = deque(maxlen=100)
            memory = self._selection_memories[list_key]

            # Create a Random object with the seed for better randomness
            rng = random.Random(seed)

            # Calculate memory size (less than half of list size)
            memory_size = max(1, len(text_list) // 2 - 1)

            # Try to select an index not in recent memory
            max_retries = min(len(text_list), 50)  # Limit retries
            selected_index = None

            for _ in range(max_retries):
                candidate_index = rng.randint(0, len(text_list) - 1)
                # Check if candidate index is in recent memory
                if candidate_index not in list(memory)[-memory_size:]:
                    selected_index = candidate_index
                    break

            # If all retries failed (unlikely), just use a random index
            if selected_index is None:
                selected_index = rng.randint(0, len(text_list) - 1)

            # Add selected index to this list's memory
            memory.append(selected_index)

            selected_text = text_list[selected_index]

        return (selected_text,)


class XJRandomTextFromFile:
    # Class-level memory to track recently selected items per unique file
    # Key: hash of file content, Value: deque of recently selected indices
    _selection_memories = {}  # Dict of deques, one per unique file

    @classmethod
    def INPUT_TYPES(s):
        # Enumerate .txt and .md files from input directory
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [
                f
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
                and (f.lower().endswith(".txt") or f.lower().endswith(".md"))
            ]
        files = sorted(files) if files else [""]

        return {
            "required": {
                "file_path": (files, {"default": files[0] if files else ""}),
                "type": (("fixed", "random"),),
                "choice": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "step": 1,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "file_path_override": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "make_list"
    CATEGORY = "XJNode/text"

    def make_list(self, file_path, type, choice, seed, file_path_override=None):
        # Use override if provided (from Text File Info node)
        if file_path_override is not None and file_path_override.strip():
            file_path = file_path_override

        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            # Read the file and split it into lines
            text_list = f.read().splitlines()
        # Remove empty lines
        text_list = [text.strip().lstrip("- ") for text in text_list]
        text_list = [
            text for text in text_list if text and not text.startswith("#")
        ]  # skip comments
        if not text_list:
            return ([""],)

        if type == "fixed":
            # If type is fixed, return the first 'choice' number of strings
            if choice >= len(text_list) + 1:
                raise Exception(f"Choice {choice} exceeded max length {len(text_list)}")
            selected_text = text_list[choice - 1]
        else:
            # Create a unique key for this file based on its content
            list_key = hashlib.md5('|'.join(text_list).encode('utf-8')).hexdigest()

            # Get or create memory deque for this specific file
            if list_key not in self._selection_memories:
                self._selection_memories[list_key] = deque(maxlen=100)
            memory = self._selection_memories[list_key]

            # Create a Random object with the seed for better randomness
            rng = random.Random(seed)

            # Calculate memory size (less than half of list size)
            memory_size = max(1, len(text_list) // 2 - 1)

            # Try to select an index not in recent memory
            max_retries = min(len(text_list), 50)  # Limit retries
            selected_index = None

            for _ in range(max_retries):
                candidate_index = rng.randint(0, len(text_list) - 1)
                # Check if candidate index is in recent memory
                if candidate_index not in list(memory)[-memory_size:]:
                    selected_index = candidate_index
                    break

            # If all retries failed (unlikely), just use a random index
            if selected_index is None:
                selected_index = rng.randint(0, len(text_list) - 1)

            # Add selected index to this file's memory
            memory.append(selected_index)

            selected_text = text_list[selected_index]

        return (selected_text,)


class XJTextListFromFile:
    """Load all valid lines from a text file as a list"""

    @classmethod
    def INPUT_TYPES(s):
        # Enumerate .txt and .md files from input directory
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [
                f
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
                and (f.lower().endswith(".txt") or f.lower().endswith(".md"))
            ]
        files = sorted(files) if files else [""]

        return {
            "required": {
                "file_path": (files, {"default": files[0] if files else ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_list"
    CATEGORY = "XJNode/text"

    def load_list(self, file_path):
        input_dir = folder_paths.get_input_directory()
        full_path = os.path.join(input_dir, file_path)

        if not os.path.exists(full_path):
            return ([],)

        # Read and process lines (same logic as XJRandomTextFromFile)
        with open(full_path, "r", encoding="utf-8") as f:
            text_list = f.read().splitlines()

        # Remove empty lines and comments
        text_list = [text.strip().lstrip("- ") for text in text_list]
        text_list = [text for text in text_list if text and not text.startswith("#")]

        # Return as list
        return (text_list,)


class XJRandomText:
    """Generate random text from specified characters"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chars": (
                    "STRING",
                    {
                        "default": string.printable.strip(),
                        "multiline": False,
                    },
                ),
                "length": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "XJNode/text"

    def generate(self, chars, length, seed):
        # Validate chars input
        if not chars:
            raise Exception("chars parameter cannot be empty")

        # Set the random seed for reproducibility
        random.seed(seed)

        # Generate random text from the specified characters
        random_text = ''.join(random.choice(chars) for _ in range(length))

        return (random_text,)


NODE_CLASS_MAPPINGS = {
    "XJRandomTextFromList": XJRandomTextFromList,
    "XJRandomTextFromFile": XJRandomTextFromFile,
    "XJTextListFromFile": XJTextListFromFile,
    "XJRandomText": XJRandomText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJRandomTextFromList": "Random Text From List",
    "XJRandomTextFromFile": "Random Text From File",
    "XJTextListFromFile": "Text List From File",
    "XJRandomText": "Random Text",
}
