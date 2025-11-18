import os
import random
import folder_paths


class XJRandomTextFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multiline_text": ("XJ_NUMBERED_LIST", {}),
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "make_list"
    CATEGORY = "XJNode/text"

    def make_list(self, multiline_text, type, choice, seed):
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
        else:
            # Set the random seed for reproducibility
            random.seed(seed)
            # Randomly select one string from the list
            selected_text = random.choice(text_list)

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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "make_list"
    CATEGORY = "XJNode/text"

    def make_list(self, file_path, type, choice, seed):
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
            # Set the random seed for reproducibility
            random.seed(seed)
            # Randomly select one string from the list
            selected_text = random.choice(text_list)

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
        text_list = [
            text for text in text_list if text and not text.startswith("#")
        ]

        # Return as list
        return (text_list,)


NODE_CLASS_MAPPINGS = {
    "XJRandomTextFromList": XJRandomTextFromList,
    "XJRandomTextFromFile": XJRandomTextFromFile,
    "XJTextListFromFile": XJTextListFromFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJRandomTextFromList": "Random Text From List",
    "XJRandomTextFromFile": "Random Text From File",
    "XJTextListFromFile": "Text List From File",
}
