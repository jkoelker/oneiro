"""Tests for long prompt support."""

from unittest.mock import Mock

import torch

from oneiro.pipelines.long_prompt import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    get_t5_tokens_and_weights,
    get_tokens_and_weights,
    get_weighted_text_embeddings_flux,
    get_weighted_text_embeddings_sd3,
    get_weighted_text_embeddings_sd15,
    get_weighted_text_embeddings_sdxl,
    group_tokens_into_chunks,
    pad_chunks_to_same_count,
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

    def test_multiple_consecutive_breaks_not_merged(self):
        """Multiple consecutive BREAKs should remain separate entries."""
        result = parse_prompt_attention("first BREAK BREAK BREAK second")
        # Count BREAK markers - should be exactly 3
        break_count = sum(1 for text, weight in result if text == "BREAK" and weight == -1)
        assert break_count == 3, f"Expected 3 BREAK markers, got {break_count}: {result}"
        # Ensure BREAKs are not merged into "BREAKBREAKBREAK"
        merged_breaks = [text for text, _ in result if "BREAKBREAK" in text]
        assert len(merged_breaks) == 0, f"BREAKs were incorrectly merged: {merged_breaks}"

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

    def test_empty_input_returns_one_chunk(self):
        """Empty token list should return one empty chunk."""
        tokens: list[int] = []
        weights: list[float] = []

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights)

        # Should return exactly one empty chunk
        assert len(chunks) == 1
        assert len(weight_chunks) == 1
        # Chunk should have correct structure: [BOS] + 75 EOS + [EOS] = 77 tokens
        assert len(chunks[0]) == 77
        assert chunks[0][0] == BOS_TOKEN_ID
        assert len(weight_chunks[0]) == 77
        assert all(w == 1.0 for w in weight_chunks[0])

    def test_only_break_markers_returns_one_chunk(self):
        """Token list with only BREAK markers should return one empty chunk."""
        # BREAK markers are represented as -1
        tokens = [-1, -1, -1]
        weights = [-1.0, -1.0, -1.0]

        chunks, weight_chunks = group_tokens_into_chunks(tokens, weights)

        # Should return exactly one empty chunk since all tokens were BREAK markers
        assert len(chunks) == 1
        assert len(weight_chunks) == 1
        # Chunk should have correct structure
        assert len(chunks[0]) == 77
        assert chunks[0][0] == BOS_TOKEN_ID


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

        # Verify tokens and weights have correct structure
        assert isinstance(tokens, list), "tokens should be a list"
        assert isinstance(weights, list), "weights should be a list"
        assert len(tokens) == len(weights), "tokens and weights should have same length"

    def test_empty_prompt(self):
        """Empty prompt should use placeholder."""
        from unittest.mock import Mock

        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, 100, EOS_TOKEN_ID])

        tokens, weights = get_tokens_and_weights(tokenizer, "")

        # Should have called tokenizer with "empty" placeholder
        tokenizer.assert_called()

        # Verify tokens and weights have correct structure
        assert isinstance(tokens, list), "tokens should be a list"
        assert isinstance(weights, list), "weights should be a list"
        assert len(tokens) == len(weights), "tokens and weights should have same length"

    def test_tokenizer_returns_insufficient_tokens(self):
        """Should handle tokenizer returning fewer than 2 tokens gracefully."""
        from unittest.mock import Mock

        tokenizer = Mock()
        # Return only BOS/EOS with no content (len < 2 after strip would be empty)
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, EOS_TOKEN_ID])

        tokens, weights = get_tokens_and_weights(tokenizer, "test")

        # Should not raise, and return valid (possibly empty) lists
        assert isinstance(tokens, list)
        assert isinstance(weights, list)
        assert len(tokens) == len(weights)

    def test_tokenizer_returns_single_token(self):
        """Should handle tokenizer returning single token gracefully."""
        from unittest.mock import Mock

        tokenizer = Mock()
        # Return only one token (unexpected edge case)
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID])

        tokens, weights = get_tokens_and_weights(tokenizer, "test")

        # Should not raise, and return valid (possibly empty) lists
        assert isinstance(tokens, list)
        assert isinstance(weights, list)
        assert len(tokens) == len(weights)

    def test_tokenizer_returns_empty_list(self):
        """Should handle tokenizer returning empty list gracefully."""
        from unittest.mock import Mock

        tokenizer = Mock()
        # Return empty input_ids (very unexpected)
        tokenizer.return_value = Mock(input_ids=[])

        tokens, weights = get_tokens_and_weights(tokenizer, "test")

        # Should not raise, and return valid (possibly empty) lists
        assert isinstance(tokens, list)
        assert isinstance(weights, list)
        assert len(tokens) == len(weights)

    def test_tokenizer_missing_input_ids_attribute(self):
        """Should handle tokenizer result missing input_ids attribute gracefully."""
        from unittest.mock import Mock

        tokenizer = Mock()
        # Return object without input_ids attribute
        result = Mock(spec=[])  # Empty spec means no attributes
        tokenizer.return_value = result

        tokens, weights = get_tokens_and_weights(tokenizer, "test")

        # Should not raise, and return valid (possibly empty) lists
        assert isinstance(tokens, list)
        assert isinstance(weights, list)
        assert len(tokens) == len(weights)


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

        # Verify tokens and weights have correct structure
        assert isinstance(tokens, list), "tokens should be a list"
        assert isinstance(weights, list), "weights should be a list"
        assert len(tokens) == len(weights), "tokens and weights should have same length"

    def test_empty_prompt(self):
        """Empty prompt should use placeholder."""
        from unittest.mock import Mock

        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[100, 1])

        tokens, weights = get_t5_tokens_and_weights(tokenizer, "")

        # Should have called tokenizer with "empty" placeholder
        tokenizer.assert_called()

        # Verify tokens and weights have correct structure
        assert isinstance(tokens, list), "tokens should be a list"
        assert isinstance(weights, list), "weights should be a list"
        assert len(tokens) == len(weights), "tokens and weights should have same length"

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

        # Verify tokens and weights have correct structure
        assert isinstance(tokens, list), "tokens should be a list"
        assert isinstance(weights, list), "weights should be a list"
        assert len(tokens) == len(weights), "tokens and weights should have same length"

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


class TestPadChunksToSameCount:
    """Tests for pad_chunks_to_same_count function."""

    def test_equal_counts_unchanged(self):
        """Equal chunk counts should return unchanged."""
        chunks_a = [[1, 2, 3]]
        weights_a = [[1.0, 1.0, 1.0]]
        chunks_b = [[4, 5, 6]]
        weights_b = [[1.0, 1.0, 1.0]]

        result = pad_chunks_to_same_count(chunks_a, weights_a, chunks_b, weights_b)

        assert result == (chunks_a, weights_a, chunks_b, weights_b)

    def test_pads_shorter_list_b(self):
        """Should pad list B when A is longer."""
        chunks_a = [[1], [2]]  # 2 chunks
        weights_a = [[1.0], [1.0]]
        chunks_b = [[3]]  # 1 chunk
        weights_b = [[1.0]]

        result_a, result_wa, result_b, result_wb = pad_chunks_to_same_count(
            chunks_a, weights_a, chunks_b, weights_b
        )

        assert len(result_a) == 2
        assert len(result_b) == 2
        # Original chunks unchanged
        assert result_a[0] == [1]
        assert result_b[0] == [3]
        # Padded chunk should be EOS-filled
        assert result_b[1][0] == BOS_TOKEN_ID

    def test_pads_shorter_list_a(self):
        """Should pad list A when B is longer."""
        chunks_a = [[1]]  # 1 chunk
        weights_a = [[1.0]]
        chunks_b = [[2], [3], [4]]  # 3 chunks
        weights_b = [[1.0], [1.0], [1.0]]

        result_a, result_wa, result_b, result_wb = pad_chunks_to_same_count(
            chunks_a, weights_a, chunks_b, weights_b
        )

        assert len(result_a) == 3
        assert len(result_b) == 3
        # Padded chunks should have correct structure
        assert result_a[1][0] == BOS_TOKEN_ID
        assert result_a[2][0] == BOS_TOKEN_ID

    def test_empty_chunk_structure(self):
        """Padded chunks should have correct BOS/EOS structure."""
        # Realistic case: B has fewer chunks than A but the same chunk length
        chunks_a = [[1] * 77, [2] * 77]  # 2 chunks
        weights_a = [[1.0] * 77, [1.0] * 77]
        chunks_b = [[3] * 77]  # 1 chunk
        weights_b = [[1.0] * 77]

        result_a, result_wa, result_b, result_wb = pad_chunks_to_same_count(
            chunks_a, weights_a, chunks_b, weights_b
        )

        # Padded chunk should be 77 tokens
        assert len(result_b[1]) == 77
        # Padded weights should be 77 values of 1.0
        assert len(result_wb[1]) == 77
        assert all(w == 1.0 for w in result_wb[1])


class TestGetWeightedTextEmbeddingsSD15:
    """Tests for get_weighted_text_embeddings_sd15 function."""

    def _create_mock_pipe(self, device: str = "cpu"):
        """Create a mock SD 1.5 pipeline."""
        pipe = Mock()
        pipe.device = device

        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, 100, 101, EOS_TOKEN_ID])
        pipe.tokenizer = tokenizer

        # Mock text encoder
        text_encoder = Mock()
        text_encoder.device = device
        text_encoder.dtype = torch.float32

        # Mock encoder output - returns embeddings with shape (batch, seq_len, hidden_dim)
        def mock_encode(input_ids):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_dim = 768  # SD 1.5 uses 768
            # Return a mock output with the expected structure
            output = Mock()
            output.__getitem__ = Mock(return_value=torch.randn(batch_size, seq_len, hidden_dim))
            return output

        text_encoder.side_effect = mock_encode
        text_encoder.__call__ = mock_encode
        pipe.text_encoder = text_encoder

        return pipe

    def test_returns_tuple_of_tensors(self):
        """Should return tuple of (prompt_embeds, negative_prompt_embeds)."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_sd15(pipe, "a cat", "bad quality")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    def test_empty_prompts(self):
        """Should handle empty prompts."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_sd15(pipe, "", "")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_uses_correct_device(self):
        """Should use pipe's device."""
        pipe = self._create_mock_pipe(device="cpu")

        prompt_embeds, neg_embeds = get_weighted_text_embeddings_sd15(pipe, "test", "negative")

        # Embeddings should be on CPU
        assert prompt_embeds.device.type == "cpu"
        assert neg_embeds.device.type == "cpu"

    def test_handles_weighted_prompt(self):
        """Should process prompts with weight syntax."""
        pipe = self._create_mock_pipe()

        # This should not raise
        result = get_weighted_text_embeddings_sd15(pipe, "a (beautiful:1.5) cat", "[ugly]")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_handles_only_break_keywords(self):
        """Should handle prompts that are only BREAK keywords (edge case).

        When input is only BREAK keywords, tokenization produces no actual tokens,
        resulting in empty chunk lists. The function should handle this gracefully
        by creating at least one valid empty chunk instead of failing at torch.cat.
        """
        pipe = self._create_mock_pipe()

        # Configure tokenizer to return only BOS/EOS (no content tokens) for BREAK-only input
        # This simulates what happens when the prompt text after parsing is empty
        pipe.tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, EOS_TOKEN_ID])

        # This should not raise - the function should handle empty chunks gracefully
        result = get_weighted_text_embeddings_sd15(pipe, "BREAK", "BREAK BREAK")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)


class TestGetWeightedTextEmbeddingsSDXL:
    """Tests for get_weighted_text_embeddings_sdxl function."""

    def _create_mock_pipe(self, device: str = "cpu"):
        """Create a mock SDXL pipeline with dual encoders."""
        pipe = Mock()
        pipe.device = device

        # Mock tokenizers (CLIP-L and CLIP-G)
        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, 100, 101, EOS_TOKEN_ID])
        pipe.tokenizer = tokenizer

        tokenizer_2 = Mock()
        tokenizer_2.return_value = Mock(input_ids=[BOS_TOKEN_ID, 200, 201, EOS_TOKEN_ID])
        pipe.tokenizer_2 = tokenizer_2

        # Mock text encoders
        text_encoder = Mock()
        text_encoder.device = device
        text_encoder.dtype = torch.float32

        def mock_encode_1(input_ids, output_hidden_states=False):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_dim = 768  # CLIP-L uses 768
            output = Mock()
            output.__getitem__ = Mock(return_value=torch.randn(batch_size, seq_len, hidden_dim))
            output.hidden_states = [
                torch.randn(batch_size, seq_len, hidden_dim)
                for _ in range(13)  # 12 layers + embedding
            ]
            return output

        text_encoder.side_effect = mock_encode_1
        text_encoder.__call__ = mock_encode_1
        pipe.text_encoder = text_encoder

        text_encoder_2 = Mock()
        text_encoder_2.device = device
        text_encoder_2.dtype = torch.float32

        def mock_encode_2(input_ids, output_hidden_states=False):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_dim = 1280  # CLIP-G uses 1280
            output = Mock()
            # Pooled output
            output.__getitem__ = Mock(return_value=torch.randn(batch_size, hidden_dim))
            output.hidden_states = [
                torch.randn(batch_size, seq_len, hidden_dim)
                for _ in range(25)  # 24 layers + embedding
            ]
            return output

        text_encoder_2.side_effect = mock_encode_2
        text_encoder_2.__call__ = mock_encode_2
        pipe.text_encoder_2 = text_encoder_2

        return pipe

    def test_returns_tuple_of_four_tensors(self):
        """Should return tuple of (prompt_embeds, neg_embeds, pooled, neg_pooled)."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_sdxl(pipe, "a cat", "bad quality")

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], torch.Tensor)  # prompt_embeds
        assert isinstance(result[1], torch.Tensor)  # negative_prompt_embeds
        assert isinstance(result[2], torch.Tensor)  # pooled_prompt_embeds
        assert isinstance(result[3], torch.Tensor)  # negative_pooled_prompt_embeds

    def test_empty_prompts(self):
        """Should handle empty prompts."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_sdxl(pipe, "", "")

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_uses_correct_device(self):
        """Should use pipe's device."""
        pipe = self._create_mock_pipe(device="cpu")

        prompt_embeds, neg_embeds, pooled, neg_pooled = get_weighted_text_embeddings_sdxl(
            pipe, "test", "negative"
        )

        assert prompt_embeds.device.type == "cpu"
        assert neg_embeds.device.type == "cpu"

    def test_handles_weighted_prompt(self):
        """Should process prompts with weight syntax."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_sdxl(pipe, "a (beautiful:1.5) cat", "[ugly]")

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_concatenates_encoder_outputs(self):
        """Should concatenate CLIP-L and CLIP-G embeddings along feature dimension."""
        pipe = self._create_mock_pipe()

        prompt_embeds, _, _, _ = get_weighted_text_embeddings_sdxl(pipe, "test", "")

        # SDXL concatenates 768 + 1280 = 2048 dimension
        assert prompt_embeds.shape[-1] == 2048

    def test_handles_only_break_keywords(self):
        """Should handle prompts that are only BREAK keywords (edge case).

        When input is only BREAK keywords, tokenization produces no actual tokens,
        resulting in empty chunk lists. The function should handle this gracefully
        by creating at least one valid empty chunk instead of failing.
        """
        pipe = self._create_mock_pipe()

        # Configure tokenizers to return only BOS/EOS (no content tokens) for BREAK-only input
        # This simulates what happens when the prompt text after parsing is empty
        pipe.tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, EOS_TOKEN_ID])
        pipe.tokenizer_2.return_value = Mock(input_ids=[BOS_TOKEN_ID, EOS_TOKEN_ID])

        # This should not raise - the function should handle empty chunks gracefully
        result = get_weighted_text_embeddings_sdxl(pipe, "BREAK", "BREAK BREAK")

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], torch.Tensor)  # prompt_embeds
        assert isinstance(result[1], torch.Tensor)  # negative_prompt_embeds
        assert isinstance(result[2], torch.Tensor)  # pooled_prompt_embeds
        assert isinstance(result[3], torch.Tensor)  # negative_pooled_prompt_embeds


class TestGetWeightedTextEmbeddingsFlux:
    """Tests for get_weighted_text_embeddings_flux function."""

    def _create_mock_pipe(self, device: str = "cpu"):
        """Create a mock Flux pipeline."""
        pipe = Mock()
        pipe.device = device

        # Mock CLIP tokenizer
        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, 100, 101, EOS_TOKEN_ID])
        pipe.tokenizer = tokenizer

        # Mock T5 tokenizer
        tokenizer_2 = Mock()
        tokenizer_2.return_value = Mock(input_ids=[100, 101, 102, 1])  # T5 format with EOS=1
        tokenizer_2.eos_token_id = 1
        pipe.tokenizer_2 = tokenizer_2

        # Mock CLIP text encoder (for pooled embeddings)
        text_encoder = Mock()
        text_encoder.device = device
        text_encoder.dtype = torch.float32

        def mock_clip_encode(input_ids, output_hidden_states=False):
            batch_size = input_ids.shape[0]
            hidden_dim = 768
            output = Mock()
            output.pooler_output = torch.randn(batch_size, hidden_dim)
            return output

        text_encoder.side_effect = mock_clip_encode
        text_encoder.__call__ = mock_clip_encode
        pipe.text_encoder = text_encoder

        # Mock T5 text encoder
        text_encoder_2 = Mock()
        text_encoder_2.device = device
        text_encoder_2.dtype = torch.float32

        def mock_t5_encode(input_ids):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_dim = 4096  # T5-XXL uses 4096
            output = [torch.randn(batch_size, seq_len, hidden_dim)]
            return output

        text_encoder_2.side_effect = mock_t5_encode
        text_encoder_2.__call__ = mock_t5_encode
        pipe.text_encoder_2 = text_encoder_2

        return pipe

    def test_returns_tuple_of_two_tensors(self):
        """Should return tuple of (prompt_embeds, pooled_prompt_embeds)."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_flux(pipe, "a cat")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)  # prompt_embeds (T5)
        assert isinstance(result[1], torch.Tensor)  # pooled_prompt_embeds (CLIP)

    def test_empty_prompt(self):
        """Should handle empty prompt."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_flux(pipe, "")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_uses_correct_device(self):
        """Should use pipe's device."""
        pipe = self._create_mock_pipe(device="cpu")

        prompt_embeds, pooled = get_weighted_text_embeddings_flux(pipe, "test")

        assert prompt_embeds.device.type == "cpu"
        assert pooled.device.type == "cpu"

    def test_handles_weighted_prompt(self):
        """Should process prompts with weight syntax."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_flux(pipe, "a (beautiful:1.5) cat")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_t5_embedding_dimension(self):
        """Should return T5 embeddings with 4096 dimension."""
        pipe = self._create_mock_pipe()

        prompt_embeds, _ = get_weighted_text_embeddings_flux(pipe, "test")

        assert prompt_embeds.shape[-1] == 4096

    def test_pooled_embedding_dimension(self):
        """Should return pooled CLIP embeddings with 768 dimension."""
        pipe = self._create_mock_pipe()

        _, pooled = get_weighted_text_embeddings_flux(pipe, "test")

        assert pooled.shape[-1] == 768

    def test_break_only_prompt_creates_fallback_chunk(self):
        """Should handle BREAK-only prompts by creating fallback empty chunk.

        When a prompt consists only of BREAK keywords, clip_chunks would be empty,
        which would cause torch.stack to fail. The function should create a fallback
        empty chunk in this case.
        """
        pipe = self._create_mock_pipe()

        # BREAK-only prompt would result in empty clip_chunks without the fix
        result = get_weighted_text_embeddings_flux(pipe, "BREAK")

        # Should not raise and should return valid tensors
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)  # prompt_embeds
        assert isinstance(result[1], torch.Tensor)  # pooled_prompt_embeds

    def test_multiple_breaks_only_creates_fallback_chunk(self):
        """Should handle multiple BREAK keywords without other content."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_flux(pipe, "BREAK BREAK BREAK")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)


class TestGetWeightedTextEmbeddingsSD3:
    """Tests for get_weighted_text_embeddings_sd3 function."""

    def _create_mock_pipe(self, device: str = "cpu", has_t5: bool = True):
        """Create a mock SD3 pipeline with triple encoders."""
        pipe = Mock()
        pipe.device = device

        # Mock CLIP tokenizers
        tokenizer = Mock()
        tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, 100, 101, EOS_TOKEN_ID])
        pipe.tokenizer = tokenizer

        tokenizer_2 = Mock()
        tokenizer_2.return_value = Mock(input_ids=[BOS_TOKEN_ID, 200, 201, EOS_TOKEN_ID])
        pipe.tokenizer_2 = tokenizer_2

        # Mock T5 tokenizer
        tokenizer_3 = Mock()
        tokenizer_3.return_value = Mock(input_ids=[300, 301, 1])  # T5 format
        tokenizer_3.eos_token_id = 1
        pipe.tokenizer_3 = tokenizer_3

        # Mock CLIP-L encoder
        text_encoder = Mock()
        text_encoder.device = device
        text_encoder.dtype = torch.float32

        def mock_encode_1(input_ids, output_hidden_states=False):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_dim = 768
            output = Mock()
            output.__getitem__ = Mock(return_value=torch.randn(batch_size, hidden_dim))
            output.hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(13)]
            return output

        text_encoder.side_effect = mock_encode_1
        text_encoder.__call__ = mock_encode_1
        pipe.text_encoder = text_encoder

        # Mock CLIP-G encoder
        text_encoder_2 = Mock()
        text_encoder_2.device = device
        text_encoder_2.dtype = torch.float32

        def mock_encode_2(input_ids, output_hidden_states=False):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_dim = 1280
            output = Mock()
            output.__getitem__ = Mock(return_value=torch.randn(batch_size, hidden_dim))
            output.hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(25)]
            return output

        text_encoder_2.side_effect = mock_encode_2
        text_encoder_2.__call__ = mock_encode_2
        pipe.text_encoder_2 = text_encoder_2

        # Mock T5 encoder (optional)
        if has_t5:
            text_encoder_3 = Mock()
            text_encoder_3.device = device
            text_encoder_3.dtype = torch.float32

            def mock_encode_3(input_ids):
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                hidden_dim = 4096
                return [torch.randn(batch_size, seq_len, hidden_dim)]

            text_encoder_3.side_effect = mock_encode_3
            text_encoder_3.__call__ = mock_encode_3
            pipe.text_encoder_3 = text_encoder_3
        else:
            pipe.text_encoder_3 = None

        return pipe

    def test_returns_tuple_of_four_tensors(self):
        """Should return tuple of (prompt_embeds, neg_embeds, pooled, neg_pooled)."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_sd3(pipe, "a cat", "bad quality")

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)
        assert isinstance(result[2], torch.Tensor)
        assert isinstance(result[3], torch.Tensor)

    def test_empty_prompts(self):
        """Should handle empty prompts."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_sd3(pipe, "", "")

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_uses_correct_device(self):
        """Should use pipe's device."""
        pipe = self._create_mock_pipe(device="cpu")

        prompt_embeds, neg_embeds, pooled, neg_pooled = get_weighted_text_embeddings_sd3(
            pipe, "test", "negative"
        )

        assert prompt_embeds.device.type == "cpu"
        assert neg_embeds.device.type == "cpu"

    def test_handles_weighted_prompt(self):
        """Should process prompts with weight syntax."""
        pipe = self._create_mock_pipe()

        result = get_weighted_text_embeddings_sd3(pipe, "a (beautiful:1.5) cat", "[ugly]")

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_pooled_embeddings_concatenated(self):
        """Should concatenate pooled embeddings from CLIP-L and CLIP-G."""
        pipe = self._create_mock_pipe()

        _, _, pooled, neg_pooled = get_weighted_text_embeddings_sd3(pipe, "test", "negative")

        # Pooled: 768 (CLIP-L) + 1280 (CLIP-G) = 2048
        assert pooled.shape[-1] == 2048
        assert neg_pooled.shape[-1] == 2048

    def test_without_t5_encoder(self):
        """Should work without T5 encoder (creates zero tensor)."""
        pipe = self._create_mock_pipe(has_t5=False)

        result = get_weighted_text_embeddings_sd3(pipe, "test", "negative")

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_handles_only_break_keywords(self):
        """Should handle prompts that are only BREAK keywords (edge case).

        When input is only BREAK keywords, tokenization produces no actual tokens,
        resulting in empty chunk lists. The function should handle this gracefully
        by creating at least one valid empty chunk instead of failing.
        """
        pipe = self._create_mock_pipe()

        # Configure CLIP tokenizers to return only BOS/EOS (no content tokens) for BREAK-only input
        # This simulates what happens when the prompt text after parsing is empty
        pipe.tokenizer.return_value = Mock(input_ids=[BOS_TOKEN_ID, EOS_TOKEN_ID])
        pipe.tokenizer_2.return_value = Mock(input_ids=[BOS_TOKEN_ID, EOS_TOKEN_ID])

        # This should not raise - the function should handle empty chunks gracefully
        result = get_weighted_text_embeddings_sd3(pipe, "BREAK", "BREAK BREAK")

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], torch.Tensor)  # prompt_embeds
        assert isinstance(result[1], torch.Tensor)  # negative_prompt_embeds
        assert isinstance(result[2], torch.Tensor)  # pooled_prompt_embeds
        assert isinstance(result[3], torch.Tensor)  # negative_pooled_prompt_embeds
