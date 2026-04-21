import json


def loads(value):
    """Parse JSON with a compatibility fallback for mildly malformed content."""
    if not isinstance(value, str):
        raise TypeError("JSON value must be a string")

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        fixed = _remove_trailing_commas(value)
        if fixed != value:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    parser = _CompatParser(value)
    result = parser.parse()
    return result


def normalize(value):
    """Return normalized JSON text, or None if the input cannot be repaired."""
    try:
        parsed = loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return json.dumps(parsed, ensure_ascii=False)


def _remove_trailing_commas(value):
    out = []
    i = 0
    in_string = False
    escaped = False
    length = len(value)

    while i < length:
        ch = value[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < length and value[j] in " \t\r\n":
                j += 1
            if j < length and value[j] in "}]":
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


class _CompatParser:
    def __init__(self, text):
        self.text = text
        self.length = len(text)
        self.index = 0

    def parse(self):
        self._skip_ws()
        value = self._parse_value()
        self._skip_ws()
        if self.index != self.length:
            raise json.JSONDecodeError(
                "Extra data", self.text, self.index
            )
        return value

    def _parse_value(self):
        self._skip_ws()
        if self.index >= self.length:
            raise json.JSONDecodeError("Expecting value", self.text, self.index)

        ch = self.text[self.index]
        if ch == "{":
            return self._parse_object()
        if ch == "[":
            return self._parse_array()
        if ch == '"':
            return self._parse_string({",", "}", "]"})
        if ch in "-0123456789":
            return self._parse_number()
        if self.text.startswith("true", self.index):
            self.index += 4
            return True
        if self.text.startswith("false", self.index):
            self.index += 5
            return False
        if self.text.startswith("null", self.index):
            self.index += 4
            return None

        raise json.JSONDecodeError("Unexpected character", self.text, self.index)

    def _parse_object(self):
        obj = {}
        self.index += 1
        self._skip_ws()
        if self._peek("}"):
            self.index += 1
            return obj

        while True:
            self._skip_ws()
            if not self._peek('"'):
                raise json.JSONDecodeError(
                    "Expecting property name enclosed in double quotes",
                    self.text,
                    self.index,
                )
            key = self._parse_string({":"})
            self._skip_ws()
            if not self._peek(":"):
                raise json.JSONDecodeError("Expecting ':'", self.text, self.index)
            self.index += 1
            value = self._parse_value()
            obj[key] = value
            self._skip_ws()
            if self._peek("}"):
                self.index += 1
                return obj
            if not self._peek(","):
                raise json.JSONDecodeError("Expecting ','", self.text, self.index)
            self.index += 1

    def _parse_array(self):
        items = []
        self.index += 1
        self._skip_ws()
        if self._peek("]"):
            self.index += 1
            return items

        while True:
            items.append(self._parse_value())
            self._skip_ws()
            if self._peek("]"):
                self.index += 1
                return items
            if not self._peek(","):
                raise json.JSONDecodeError("Expecting ','", self.text, self.index)
            self.index += 1

    def _parse_string(self, terminators):
        self.index += 1
        out = []
        while self.index < self.length:
            ch = self.text[self.index]
            if ch == "\\":
                if self.index + 1 >= self.length:
                    raise json.JSONDecodeError(
                        "Unterminated escape sequence",
                        self.text,
                        self.index,
                    )
                escaped = self.text[self.index + 1]
                out.append(_decode_escape(escaped))
                self.index += 2
                continue

            if ch == '"':
                next_char = self._next_non_ws(self.index + 1)
                if next_char is None or next_char in terminators:
                    self.index += 1
                    return "".join(out)

            out.append(ch)
            self.index += 1

        raise json.JSONDecodeError("Unterminated string", self.text, self.index)

    def _parse_number(self):
        start = self.index
        if self._peek("-"):
            self.index += 1

        self._consume_digits()
        if self._peek("."):
            self.index += 1
            self._consume_digits()

        if self._peek("e") or self._peek("E"):
            self.index += 1
            if self._peek("+") or self._peek("-"):
                self.index += 1
            self._consume_digits()

        raw = self.text[start:self.index]
        if "." in raw or "e" in raw or "E" in raw:
            return float(raw)
        return int(raw)

    def _consume_digits(self):
        start = self.index
        while self.index < self.length and self.text[self.index].isdigit():
            self.index += 1
        if self.index == start:
            raise json.JSONDecodeError("Expecting digits", self.text, self.index)

    def _skip_ws(self):
        while self.index < self.length and self.text[self.index] in " \t\r\n":
            self.index += 1

    def _peek(self, value):
        return self.index < self.length and self.text[self.index] == value

    def _next_non_ws(self, start):
        i = start
        while i < self.length and self.text[i] in " \t\r\n":
            i += 1
        if i >= self.length:
            return None
        return self.text[i]


def _decode_escape(ch):
    mapping = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    return mapping.get(ch, ch)
