import os
import random
import string
import folder_paths


def parse_text_blocks(text):
    """
    Parse text into blocks based on line prefixes:
    - Lines starting with '-' begin a new item block
    - Lines starting with '#' begin a comment block (ignored)
    - Other lines continue the previous block

    Returns a list of text blocks (comment blocks are excluded)
    """
    lines = text.splitlines()
    blocks = []
    current_block = None
    current_type = None  # 'item' or 'comment'

    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith('-'):
            # Start new item block
            if current_block is not None and current_type == 'item':
                blocks.append('\n'.join(current_block))

            # Remove leading '- ' or '-' from first line
            if stripped.startswith('- '):
                first_line = stripped[2:]
            else:
                first_line = stripped[1:]

            current_block = [first_line] if first_line else []
            current_type = 'item'

        elif stripped.startswith('#'):
            # Start new comment block (will be ignored)
            if current_block is not None and current_type == 'item':
                blocks.append('\n'.join(current_block))

            current_block = []
            current_type = 'comment'

        else:
            # Continuation line
            if current_block is not None:
                # Preserve line as-is (including indentation)
                current_block.append(line)
            # If no current block, ignore orphan lines at the start

    # Add final block if it's an item
    if current_block is not None and current_type == 'item':
        blocks.append('\n'.join(current_block))

    # Filter out empty blocks
    blocks = [b.strip() for b in blocks if b.strip()]

    return blocks


class XJRandomTextFromList:
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
        parts = index_string.split(",")

        for part in parts:
            part = part.strip()
            if "-" in part:
                # Handle range
                range_parts = part.split("-")
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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "make_list"
    CATEGORY = "XJNodes/text"

    def make_list(self, multiline_text, type, choice, seed, index_list="1, 2, 3"):
        # Parse text into blocks (supports multiline items)
        text_list = parse_text_blocks(multiline_text)
        if not text_list:
            return ([""],)

        if type == "fixed":
            # If type is fixed, return the item at 'choice' index with wrap-around
            selected_text = text_list[(choice - 1) % len(text_list)]
        elif type == "list":
            # Parse the index list
            try:
                valid_indices = self.parse_index_list(index_list)
            except (ValueError, IndexError) as e:
                raise Exception(f"Invalid index_list format: {index_list}. Error: {e}")

            # Filter to only valid indices (within bounds)
            valid_indices = [i for i in valid_indices if 1 <= i <= len(text_list)]

            if not valid_indices:
                raise Exception(
                    f"No valid indices in range for list with {len(text_list)} items"
                )

            # Create a Random object with the seed for better randomness
            rng = random.Random(seed)

            # Convert to 0-indexed for accessing text_list
            available_indices = [i - 1 for i in valid_indices]

            selected_index = rng.choice(available_indices)
            selected_text = text_list[selected_index]
        else:  # random
            # Create a Random object with the seed for better randomness
            rng = random.Random(seed)
            selected_index = rng.randint(0, len(text_list) - 1)
            selected_text = text_list[selected_index]

        return (selected_text,)


class XJRandomTextFromFile:
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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "make_list"
    CATEGORY = "XJNodes/text"

    def make_list(self, file_path, type, choice, seed, file_path_override=None):
        # Use override if provided (from Text File Info node)
        if file_path_override is not None and file_path_override.strip():
            file_path = file_path_override

        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse text into blocks (supports multiline items)
        text_list = parse_text_blocks(content)
        if not text_list:
            return ([""],)

        if type == "fixed":
            # If type is fixed, return the item at 'choice' index with wrap-around
            selected_text = text_list[(choice - 1) % len(text_list)]
        else:
            # Create a Random object with the seed for better randomness
            rng = random.Random(seed)
            selected_index = rng.randint(0, len(text_list) - 1)
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
    CATEGORY = "XJNodes/text"

    def load_list(self, file_path):
        input_dir = folder_paths.get_input_directory()
        full_path = os.path.join(input_dir, file_path)

        if not os.path.exists(full_path):
            return ([],)

        # Read and parse text into blocks (supports multiline items)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        text_list = parse_text_blocks(content)

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
    CATEGORY = "XJNodes/text"

    def generate(self, chars, length, seed):
        # Validate chars input
        if not chars:
            raise Exception("chars parameter cannot be empty")

        # Set the random seed for reproducibility
        random.seed(seed)

        # Generate random text from the specified characters
        random_text = "".join(random.choice(chars) for _ in range(length))

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
