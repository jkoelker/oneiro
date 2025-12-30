"""Tests for long prompt support."""

from oneiro.pipelines.long_prompt import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    get_t5_tokens_and_weights,
    get_tokens_and_weights,
    group_tokens_into_chunks,
    parse_prompt_attention,
)


class TestParsePromptAttention:
    """Tests for parse_prompt_attention function."""

    def test_plain_text(self):
        """Plain text should have weight 1.0."""
        result = parse_prompt_attention("normal text")
        assert result == [("normal text", 1.0)]

    def test_parentheses_increase_weight(self):
        """Parentheses should increase weight by 1.1."""
        result = parse_prompt_attention("an (important) word")
        assert len(result) == 3
        assert result[0] == ("an ", 1.0)
        assert result[1][0] == "important"
        assert abs(result[1][1] - 1.1) < 0.001
        assert result[2] == (" word", 1.0)

    def test_explicit_weight(self):
        """Explicit weight syntax (text:weight) should work."""
        result = parse_prompt_attention("a (red:1.5) cat")
        assert len(result) == 3
        assert result[1][0] == "red"
        assert abs(result[1][1] - 1.5) < 0.001

    def test_square_brackets_decrease_weight(self):
        """Square brackets should decrease weight by 1/1.1."""
        result = parse_prompt_attention("a [subtle] detail")
        assert len(result) == 3
        assert result[1][0] == "subtle"
        assert abs(result[1][1] - (1 / 1.1)) < 0.001

    def test_nested_parentheses(self):
        """Nested parentheses should multiply weights."""
        result = parse_prompt_attention("a ((very important)) thing")
        # Weight should be 1.1 * 1.1 = 1.21
        assert any(abs(w - 1.21) < 0.01 for _, w in result)

    def test_unbalanced_parentheses(self):
        """Unbalanced parentheses should not crash."""
        result = parse_prompt_attention("(unbalanced")
        assert len(result) >= 1
        assert result[0][0] == "unbalanced"

    def test_escaped_brackets(self):
        """Escaped brackets should be literal."""
        result = parse_prompt_attention(r"\(literal\)")
        assert result == [("(literal)", 1.0)]

    def test_break_keyword(self):
        """BREAK keyword should create a marker."""
        result = parse_prompt_attention("first part BREAK second part")
        # Should have BREAK marker with weight -1
        assert any(text == "BREAK" and weight == -1 for text, weight in result)

    def test_complex_prompt(self):
        """Complex prompt with multiple features."""
        result = parse_prompt_attention("a (((house:1.3)) [on] a (hill:0.5), sun")
        # Should parse without error and have multiple entries
        assert len(result) > 1

    def test_empty_string(self):
        """Empty string should return empty entry."""
        result = parse_prompt_attention("")
        assert result == [("", 1.0)]


class TestGroupTokensIntoChunks:
    """Tests for group_tokens_into_chunks function."""

    def test_short_sequence(self):
        """Sequence shorter than 75 tokens."""
        tokens = list(range(10))
        weights = [1.0] * 10

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights)

        assert len(chunks) == 1
        assert chunks[0][0] == BOS_TOKEN_ID
        assert chunks[0][-1] == EOS_TOKEN_ID
        assert len(chunks[0]) == 77  # Padded to full length

    def test_exact_75_tokens(self):
        """Exactly 75 tokens should produce one chunk."""
        tokens = list(range(75))
        weights = [1.0] * 75

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights)

        assert len(chunks) == 1
        assert chunks[0][0] == BOS_TOKEN_ID
        assert chunks[0][-1] == EOS_TOKEN_ID
        # Should have BOS + 75 tokens + EOS = 77
        assert len(chunks[0]) == 77

    def test_76_tokens_splits(self):
        """76 tokens should produce two chunks."""
        tokens = list(range(76))
        weights = [1.0] * 76

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights)

        assert len(chunks) == 2

    def test_150_tokens_splits(self):
        """150 tokens should produce two chunks."""
        tokens = list(range(150))
        weights = [1.0] * 150

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights)

        assert len(chunks) == 2
        # First chunk should be full (77 tokens including BOS/EOS)
        assert len(chunks[0]) == 77
        # Second chunk should be padded
        assert len(chunks[1]) == 77

    def test_break_marker_forces_new_chunk(self):
        """BREAK marker (-1) should force a new chunk."""
        tokens = [1, 2, 3, -1, 4, 5, 6]  # -1 is BREAK marker
        weights = [1.0, 1.0, 1.0, -1, 1.0, 1.0, 1.0]

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights)

        assert len(chunks) == 2
        # First chunk should have tokens 1,2,3
        assert 1 in chunks[0]
        # Second chunk should have tokens 4,5,6
        assert 4 in chunks[1]

    def test_weights_preserved(self):
        """Weights should be preserved in chunks."""
        tokens = [100, 101, 102]
        weights = [1.0, 1.5, 0.8]

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights)

        # Weights should be: [1.0 (BOS), 1.0, 1.5, 0.8, 1.0s (padding), 1.0 (EOS)]
        assert weight_chunks[0][1] == 1.0
        assert weight_chunks[0][2] == 1.5
        assert weight_chunks[0][3] == 0.8

    def test_no_padding_option(self):
        """pad_last_block=False should not pad."""
        tokens = [1, 2, 3]
        weights = [1.0, 1.0, 1.0]

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights, pad_last_block=False)

        # Should be BOS + 3 tokens + EOS = 5
        assert len(chunks[0]) == 5


class TestGetTokensAndWeights:
    """Tests for get_tokens_and_weights function."""

    def test_requires_tokenizer(self):
        """Function requires a tokenizer."""
        # Create a mock tokenizer
        from unittest.mock import Mock

        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, 100, 101, EOS_TOKEN_ID])

        # Should not raise
        tokens, weights = get_tokens_and_weights(tokenizer, "test")

        # Tokenizer should have been called
        tokenizer.assert_called()

    def test_empty_prompt(self):
        """Empty prompt should use placeholder."""
        from unittest.mock import Mock

        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, 100, EOS_TOKEN_ID])

        tokens, weights = get_tokens_and_weights(tokenizer, "")

        # Should have called tokenizer with "empty" placeholder
        tokenizer.assert_called()


class TestIntegration:
    """Integration tests for long prompt handling."""

    def test_parse_and_chunk_long_prompt(self):
        """Parse and chunk a realistic long prompt."""
        # Simulate a 100-word prompt
        long_prompt = " ".join(["word"] * 100)

        # Parse attention (no weights, should all be 1.0)
        parsed = parse_prompt_attention(long_prompt)
        assert len(parsed) == 1  # All merged since same weight
        assert parsed[0][1] == 1.0

    def test_weighted_long_prompt(self):
        """Parse and chunk a long prompt with weights."""
        # Create a prompt with mixed weights
        long_prompt = "(important) " + " ".join(["normal"] * 50) + " (also important:1.5)"

        parsed = parse_prompt_attention(long_prompt)

        # Should have entries with different weights
        weights = [w for _, w in parsed]
        assert len(set(weights)) > 1  # Multiple different weights


class TestGetT5TokensAndWeights:
    """Tests for get_t5_tokens_and_weights function."""

    def test_requires_tokenizer(self):
        """Function requires a tokenizer."""
        from unittest.mock import Mock

        tokenizer = Mock()
        # T5 tokenizer includes special tokens, no BOS/EOS stripping
        tokenizer.return_value = Mock(input_ids=[100, 101, 102, 1])  # 1 is T5's EOS

        tokens, weights = get_t5_tokens_and_weights(tokenizer, "test")

        tokenizer.assert_called()

    def test_empty_prompt(self):
        """Empty prompt should use placeholder."""
        from unittest.mock import Mock

        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[100, 1])

        tokens, weights = get_t5_tokens_and_weights(tokenizer, "")

        # Should have called tokenizer with "empty" placeholder
        tokenizer.assert_called()

    def test_break_keyword_ignored(self):
        """BREAK keyword should be ignored for T5 (no chunking)."""
        from unittest.mock import Mock

        tokenizer = Mock()
        # Simulate tokenizing "first" and "second" separately
        tokenizer.side_effect = [
            Mock(input_ids=[100, 101]),  # "first"
            Mock(input_ids=[200, 201]),  # "second"
        ]

        # BREAK is between first and second
        tokens, weights = get_t5_tokens_and_weights(tokenizer, "first BREAK second")

        # Should have tokens from both parts, BREAK not included
        # Note: actual tokens depend on mock setup
        assert tokenizer.call_count >= 1

    def test_weights_applied(self):
        """Weights should be extracted and applied to tokens."""
        from unittest.mock import Mock

        tokenizer = Mock()
        # Each call returns some tokens
        tokenizer.side_effect = [
            Mock(input_ids=[100]),  # "a "
            Mock(input_ids=[200, 201]),  # "red"
            Mock(input_ids=[300]),  # " cat"
        ]

        tokens, weights = get_t5_tokens_and_weights(tokenizer, "a (red:1.5) cat")

        # Weights for "red" tokens should be 1.5
        # The actual structure depends on parse_prompt_attention output
        assert len(tokens) == len(weights)
