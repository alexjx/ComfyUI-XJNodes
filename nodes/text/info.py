import os
import folder_paths


class XJTextFileInfo:
    """Get metadata about a text file (line count, etc.)"""

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

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("line_count", "file_name")
    FUNCTION = "get_info"
    CATEGORY = "XJNode/text"

    def get_info(self, file_path):
        input_dir = folder_paths.get_input_directory()
        full_path = os.path.join(input_dir, file_path)

        if not os.path.exists(full_path):
            return (0, file_path)

        # Read and count valid lines (same logic as XJRandomTextFromFile)
        with open(full_path, "r", encoding="utf-8") as f:
            text_list = f.read().splitlines()

        # Remove empty lines and comments
        text_list = [text.strip().lstrip("- ") for text in text_list]
        text_list = [text for text in text_list if text and not text.startswith("#")]

        line_count = len(text_list)

        return (line_count, file_path)


NODE_CLASS_MAPPINGS = {
    "XJTextFileInfo": XJTextFileInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJTextFileInfo": "Text File Info",
}
