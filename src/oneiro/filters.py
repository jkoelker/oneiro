"""Content filtering for prompts with configurable blacklist."""

import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oneiro.config import Config


class ContentFilter:
    """Content filtering for prompts using word-based blacklist.

    Checks prompts against a configurable blacklist from config.
    Optionally allows blacklisted words in negative prompts.
    """

    def __init__(self, config: "Config | None" = None):
        self.config = config

    def check(self, prompt: str, negative_prompt: str = "") -> tuple[bool, str | None]:
        """Check prompt against blacklist.

        Args:
            prompt: The main prompt to check
            negative_prompt: The negative prompt (optionally checked)

        Returns:
            Tuple of (allowed, blocked_word). If allowed is True, blocked_word is None.
            If allowed is False, blocked_word contains the matched word.
        """
        if self.config is None:
            return True, None

        # Get blacklist configuration
        blacklist = self.config.get("blacklist", "words", default=[])
        allow_in_negative = self.config.get("blacklist", "allow_in_negative", default=True)

        if not blacklist:
            return True, None

        # Determine which text to search
        search_text = prompt if allow_in_negative else f"{prompt} {negative_prompt}"

        # Normalize: remove punctuation, lowercase, split into words
        translator = str.maketrans("", "", string.punctuation)
        words = search_text.translate(translator).lower().split()

        # Check each banned word
        for banned in blacklist:
            banned_lower = banned.lower()
            if banned_lower in words:
                return False, banned

        return True, None
