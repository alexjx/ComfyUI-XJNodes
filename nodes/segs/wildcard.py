"""
SEGS wildcard prompt parsing - extract prompts from Impact Pack wildcard syntax.
"""

import random
import re


def parse_sep_pattern(text):
    """
    Split text by [SEP], [SEP:R], or [SEP:SEED] separators.

    Returns:
        list of (seed, prompt_text) tuples
    """
    # Pattern: [SEP] or [SEP:R] or [SEP:NUMBER]
    sep_pattern = r"\[SEP(?::(\w+))?\]"

    # Split by separator while capturing the seed spec
    parts = re.split(sep_pattern, text)

    # Parts will be: [text0, seed1, text1, seed2, text2, ...]
    # Where seed can be None (no seed), 'R' (random), or a number
    results = []

    i = 0
    while i < len(parts):
        if i == 0:
            # First part (before any [SEP])
            prompt = parts[i].strip()
            if prompt:
                results.append((None, prompt))
            i += 1
        else:
            # Pattern: seed_spec, prompt_text
            if i + 1 < len(parts):
                seed_spec = parts[i]
                prompt = parts[i + 1].strip()

                if seed_spec == "R":
                    seed = "random"
                elif seed_spec and seed_spec.isdigit():
                    seed = int(seed_spec)
                else:
                    seed = None

                if prompt:
                    results.append((seed, prompt))
            i += 2

    return results


def parse_ordering_mode(text):
    """
    Parse ordering mode prefix and return (mode, remaining_text).

    Supported modes: ASC, DSC, ASC-SIZE, DSC-SIZE, RND, LAB
    """
    mode_pattern = r"^\[(ASC-SIZE|DSC-SIZE|ASC|DSC|RND|LAB)\]"
    match = re.match(mode_pattern, text)

    if match:
        mode = match.group(1)
        remaining = text[len(match.group(0)) :]
        return mode, remaining

    return None, text


def parse_label_mode(text):
    """
    Parse [LAB] mode syntax: [LAB][label1]text1[label2]text2[ALL]common_text

    Returns:
        dict mapping label -> text
    """
    # Remove [LAB] prefix
    if text.startswith("[LAB]"):
        text = text[5:]

    # Pattern: [Label]content (until next [ or end)
    pattern = r"\[([A-Za-z0-9_. ]+)\]([^\[]+)"
    matches = re.findall(pattern, text)

    label_dict = {}
    for label, content in matches:
        label = label.strip()
        content = content.strip()
        if label in label_dict:
            label_dict[label] += " " + content
        else:
            label_dict[label] = content

    return label_dict


def expand_options(text):
    """
    Expand {option1|option2|option3} syntax with random selection.

    Supports:
    - Basic: {a|b|c}
    - Weighted: {3::a|2::b|c} (weights 3:2:1)
    - Multi-select: {2$$,$$a|b|c|d} (select 2, join with comma)
    """
    # Pattern for options: {content}
    option_pattern = r"\{([^{}]+)\}"

    def replace_option(match):
        content = match.group(1)

        # Check for multi-select: N$$sep$$options
        multi_match = re.match(r"(\d+)\$\$(.+?)\$\$(.+)", content)
        if multi_match:
            count = int(multi_match.group(1))
            separator = multi_match.group(2)
            options_str = multi_match.group(3)
            options = [opt.strip() for opt in options_str.split("|")]
            selected = random.sample(options, min(count, len(options)))
            return separator.join(selected)

        # Split options
        options = content.split("|")

        # Check for weighted options: N::option
        weighted = []
        for opt in options:
            weight_match = re.match(r"(\d+)::(.+)", opt.strip())
            if weight_match:
                weight = int(weight_match.group(1))
                option_text = weight_match.group(2).strip()
                weighted.extend([option_text] * weight)
            else:
                weighted.append(opt.strip())

        return random.choice(weighted) if weighted else ""

    # Replace all option patterns
    expanded = re.sub(option_pattern, replace_option, text)
    return expanded


class XJSegsWildcardPrompt:
    """
    Parse Impact Pack wildcard syntax and extract prompt for a specific segment index.

    Supports:
    - [ASC], [DSC], [ASC-SIZE], [DSC-SIZE], [RND] ordering modes
    - [SEP], [SEP:R], [SEP:SEED] separators
    - [LAB][label]text label-based mode
    - {option1|option2} basic option expansion
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_text": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000}),
            },
            "optional": {
                "label": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    CATEGORY = "XJNodes/segs"
    FUNCTION = "parse"

    def parse(self, wildcard_text, index, label="", seed=0):
        """
        Parse wildcard text and return prompt for the given index.

        Args:
            wildcard_text: Wildcard syntax text
            index: Segment index (0-based)
            label: Segment label (for [LAB] mode)
            seed: Base seed for randomization

        Returns:
            Extracted prompt string
        """
        if not wildcard_text:
            return ("",)

        # Parse ordering mode
        mode, remaining_text = parse_ordering_mode(wildcard_text)

        # Handle [LAB] mode
        if mode == "LAB":
            label_dict = parse_label_mode(wildcard_text)

            # Build prompt from label dict
            prompt = ""
            if "ALL" in label_dict:
                prompt = label_dict["ALL"]

            if label and label in label_dict:
                if prompt:
                    prompt += " " + label_dict[label]
                else:
                    prompt = label_dict[label]

            # Expand options
            prompt = expand_options(prompt)
            return (prompt.strip(),)

        # Parse [SEP] patterns
        if mode in ["ASC", "DSC", "ASC-SIZE", "DSC-SIZE", "RND", None]:
            prompts = parse_sep_pattern(remaining_text)

            if not prompts:
                # No [SEP] found, treat entire text as single prompt
                prompts = [(None, remaining_text)]

            # Handle RND mode - shuffle prompts
            if mode == "RND":
                random.seed(seed)
                random.shuffle(prompts)

            # Get prompt for this index (cycle if index >= len)
            if prompts:
                idx = index % len(prompts)
                prompt_seed, prompt_text = prompts[idx]

                # Set random seed if specified
                if prompt_seed == "random":
                    random.seed(seed + index)
                elif isinstance(prompt_seed, int):
                    random.seed(prompt_seed)

                # Expand options
                prompt_text = expand_options(prompt_text)
                return (prompt_text.strip(),)

        return ("",)


NODE_CLASS_MAPPINGS = {
    "XJSegsWildcardPrompt": XJSegsWildcardPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSegsWildcardPrompt": "SEGS Wildcard Prompt",
}
