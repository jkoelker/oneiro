"""Tests for ContentFilter."""

from unittest.mock import Mock

import pytest

from oneiro.filters import ContentFilter


class TestContentFilterNoConfig:
    """Tests for ContentFilter when config is None."""

    def test_check_returns_true_when_no_config(self):
        """When config is None, all prompts should be allowed."""
        filter_ = ContentFilter(config=None)
        allowed, blocked_word = filter_.check("any prompt with banned words")
        assert allowed is True
        assert blocked_word is None


class TestContentFilterEmptyBlacklist:
    """Tests for ContentFilter with empty blacklist."""

    def test_check_allows_all_when_blacklist_empty(self):
        """When blacklist is empty, all prompts should be allowed."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): [],
                ("blacklist", "allow_in_negative"): True,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check("any prompt")
        assert allowed is True
        assert blocked_word is None


class TestContentFilterBasicBlocking:
    """Tests for basic word blocking functionality."""

    @pytest.mark.parametrize(
        "prompt,blacklist,expected_allowed,expected_word",
        [
            # Basic blocking
            ("this is banned content", ["banned"], False, "banned"),
            ("hello world", ["banned"], True, None),
            # Case insensitive
            ("this is BANNED content", ["banned"], False, "banned"),
            ("this is Banned content", ["banned"], False, "banned"),
            ("this is bAnNeD content", ["banned"], False, "banned"),
            # Punctuation handling
            ("banned!", ["banned"], False, "banned"),
            ("hello, banned.", ["banned"], False, "banned"),
            ("(banned)", ["banned"], False, "banned"),
            ("banned?!?", ["banned"], False, "banned"),
            # Multiple blacklist words - first match returned
            ("forbidden and banned", ["banned", "forbidden"], False, "banned"),
            ("forbidden only", ["banned", "forbidden"], False, "forbidden"),
            # Partial word should NOT match
            ("banner is fine", ["ban"], True, None),
            ("unbanned content", ["banned"], True, None),
            ("prebanned stuff", ["banned"], True, None),
            # Multiple words in prompt
            ("hello world", ["hello"], False, "hello"),
            ("multiple banned words here", ["words"], False, "words"),
            # Whitespace handling
            ("  banned  ", ["banned"], False, "banned"),
            ("banned   content", ["banned"], False, "banned"),
        ],
        ids=[
            "basic_block",
            "no_match",
            "case_upper",
            "case_title",
            "case_mixed",
            "punctuation_exclaim",
            "punctuation_comma_period",
            "punctuation_parens",
            "punctuation_multiple",
            "multi_blacklist_first",
            "multi_blacklist_second",
            "partial_word_banner",
            "partial_word_prefix_un",
            "partial_word_prefix_pre",
            "match_first_word",
            "match_middle_word",
            "whitespace_padded",
            "whitespace_internal",
        ],
    )
    def test_check_prompt_blocking(self, prompt, blacklist, expected_allowed, expected_word):
        """Test various prompt blocking scenarios."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): blacklist,
                ("blacklist", "allow_in_negative"): True,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check(prompt)
        assert allowed is expected_allowed
        assert blocked_word == expected_word


class TestContentFilterNegativePrompt:
    """Tests for negative prompt handling."""

    def test_allow_in_negative_true_ignores_negative_prompt(self):
        """When allow_in_negative=True, banned words in negative prompt are ignored."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): ["banned"],
                ("blacklist", "allow_in_negative"): True,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check(
            prompt="clean prompt",
            negative_prompt="banned word here",
        )
        assert allowed is True
        assert blocked_word is None

    def test_allow_in_negative_false_checks_negative_prompt(self):
        """When allow_in_negative=False, banned words in negative prompt are blocked."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): ["banned"],
                ("blacklist", "allow_in_negative"): False,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check(
            prompt="clean prompt",
            negative_prompt="banned word here",
        )
        assert allowed is False
        assert blocked_word == "banned"

    def test_allow_in_negative_false_blocks_prompt_too(self):
        """When allow_in_negative=False, banned words in main prompt are still blocked."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): ["banned"],
                ("blacklist", "allow_in_negative"): False,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check(
            prompt="banned content",
            negative_prompt="clean negative",
        )
        assert allowed is False
        assert blocked_word == "banned"


class TestContentFilterEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_prompt(self):
        """Empty prompt should be allowed."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): ["banned"],
                ("blacklist", "allow_in_negative"): True,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check("")
        assert allowed is True
        assert blocked_word is None

    def test_unicode_in_prompt(self):
        """Unicode characters in prompt should be handled."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): ["banned"],
                ("blacklist", "allow_in_negative"): True,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check("hëllo wörld 日本語")
        assert allowed is True
        assert blocked_word is None

    def test_unicode_in_blacklist(self):
        """Unicode characters in blacklist should match."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): ["日本語"],
                ("blacklist", "allow_in_negative"): True,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check("hello 日本語 world")
        assert allowed is False
        assert blocked_word == "日本語"

    def test_numbers_in_prompt(self):
        """Numbers should not interfere with word matching."""
        config = Mock()
        config.get = Mock(
            side_effect=lambda *keys, default=None: {
                ("blacklist", "words"): ["banned"],
                ("blacklist", "allow_in_negative"): True,
            }.get(keys, default)
        )
        filter_ = ContentFilter(config=config)
        allowed, blocked_word = filter_.check("test 123 banned 456")
        assert allowed is False
        assert blocked_word == "banned"

    def test_uses_mock_config_fixture(self, mock_config):
        """Test using the mock_config fixture from conftest."""
        filter_ = ContentFilter(config=mock_config)
        # mock_config has ["banned", "forbidden"] as blacklist
        allowed, blocked_word = filter_.check("this has a banned word")
        assert allowed is False
        assert blocked_word == "banned"
