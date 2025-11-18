import json


class XJJSONExtractor:
    """
    Extract values from JSON metadata using dot notation path.

    Supports:
    - Object keys: workflow.settings.cfg
    - Array indices: nodes.0.seed
    - Nested paths: data.items.2.name
    - Nested JSON strings: Automatically parses JSON strings when more path components remain

    Returns value converted to all 4 types, or None if conversion fails.
    Raises exception if nested JSON string parsing fails.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True}),
                "key_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("string", "float", "int", "boolean")
    FUNCTION = "extract"
    CATEGORY = "XJNode/text"

    def extract(self, json_string, key_path):
        """
        Extract value from JSON and convert to all types.
        Returns tuple of (string, float, int, boolean) or None for failed conversions.
        """
        # Step 1: Parse JSON
        try:
            data = json.loads(json_string)
        except Exception:
            return (None, None, None, None)

        # Step 2 & 3: Navigate path and extract value
        value = self._navigate_path(data, key_path)
        if value is None:
            return (None, None, None, None)

        # Step 4: Convert to all types
        str_val = self._to_string(value)
        float_val = self._to_float(value)
        int_val = self._to_int(value)
        bool_val = self._to_bool(value)

        return (str_val, float_val, int_val, bool_val)

    def _navigate_path(self, data, key_path):
        """
        Navigate nested JSON using dot notation.
        Automatically parses nested JSON strings if more path components remain.

        Examples:
          - "workflow.steps" → data["workflow"]["steps"]
          - "nodes.0.seed" → data["nodes"][0]["seed"]
          - "data.items.2.name" → data["data"]["items"][2]["name"]
          - Nested JSON: "metadata.settings.cfg" where metadata is a JSON string

        Returns extracted value or None if path is invalid.
        Raises exception if nested JSON parsing fails.
        """
        if not key_path or not key_path.strip():
            return None

        keys = key_path.split(".")
        current = data

        for i, key in enumerate(keys):
            # Try as object key first (works for both string and numeric keys)
            if isinstance(current, dict) and key in current:
                current = current[key]
            # If not found in dict, try as array index (if key is numeric)
            elif key.isdigit() and isinstance(current, list):
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                return None

            # After navigating, check if we need to parse nested JSON
            # Only if: 1) we have more keys to process, 2) current value is a string
            remaining_keys = keys[i + 1:]
            if remaining_keys and isinstance(current, str):
                # Try to parse as JSON
                try:
                    current = json.loads(current)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse nested JSON at path '{'.'.join(keys[:i+1])}'. "
                        f"Expected JSON but got: {current[:100]}... "
                        f"Error: {str(e)}"
                    )

        return current

    def _to_string(self, value):
        """Convert to string or None"""
        try:
            return str(value)
        except Exception:
            return None

    def _to_float(self, value):
        """Convert to float or None"""
        try:
            return float(value)
        except Exception:
            return None

    def _to_int(self, value):
        """Convert to int or None (handles floats like 7.5 → 7)"""
        try:
            return int(float(value))
        except Exception:
            return None

    def _to_bool(self, value):
        """Convert to bool or None"""
        try:
            return bool(value)
        except Exception:
            return None


NODE_CLASS_MAPPINGS = {
    "XJJSONExtractor": XJJSONExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJJSONExtractor": "JSON Extractor",
}
