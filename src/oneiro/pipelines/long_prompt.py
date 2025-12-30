"""Long prompt support for CLIP-based pipelines.

Implements prompt chunking and embedding concatenation to support prompts longer than
CLIP's 77-token limit. Also supports A1111-style weight syntax like (word:1.5) and [word].

Based on techniques from sd_embed and lpw_stable_diffusion community pipelines.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass  # CLIPTokenizer type hints use Any for flexibility


# CLIP special tokens
BOS_TOKEN_ID = 49406  # Start of text
EOS_TOKEN_ID = 49407  # End of text
MAX_TOKENS_PER_CHUNK = 75  # 77 - 2 (BOS + EOS)

# Regex for parsing A1111-style attention weights
RE_ATTENTION = re.compile(
    r"""
    \\\(|           # escaped (
    \\\)|           # escaped )
    \\\[|           # escaped [
    \\]|            # escaped ]
    \\\\|           # escaped \
    \\|             # single escape
    \(|             # opening (
    \[|             # opening [
    :([+-]?[.\d]+)\)|  # weight like :1.5)
    \)|             # closing )
    ]|              # closing ]
    [^\\()\[\]:]+|  # regular text
    :               # colon
    """,
    re.X,
)

# BREAK keyword splits prompt into separate chunks
RE_BREAK = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str) -> list[tuple[str, float]]:
    """Parse prompt with A1111-style attention weights.

    Supported syntax:
      (abc) - increases attention by 1.1x
      (abc:1.5) - increases attention by 1.5x
      [abc] - decreases attention by 1.1x (same as :0.909)
      \\( \\) \\[ \\] - literal brackets
      BREAK - forces a new chunk boundary

    Args:
        text: Prompt text with optional weight syntax

    Returns:
        List of (text, weight) tuples
    """
    res: list[list[Any]] = []
    round_brackets: list[int] = []
    square_brackets: list[int] = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float) -> None:
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in RE_ATTENTION.finditer(text):
        match_text = m.group(0)
        weight = m.group(1)

        if match_text.startswith("\\"):
            res.append([match_text[1:], 1.0])
        elif match_text == "(":
            round_brackets.append(len(res))
        elif match_text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif match_text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif match_text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            # Handle BREAK keyword
            parts = re.split(RE_BREAK, match_text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                if part:
                    res.append([part, 1.0])

    # Handle unbalanced brackets
    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # Merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return [(text, weight) for text, weight in res]


def get_tokens_and_weights(
    tokenizer: Any,
    prompt: str,
) -> tuple[list[int], list[float]]:
    """Tokenize prompt and extract per-token weights.

    Args:
        tokenizer: CLIP tokenizer
        prompt: Prompt text with optional weight syntax

    Returns:
        Tuple of (token_ids, weights) lists
    """
    if not prompt or len(prompt) == 0:
        prompt = "empty"

    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens: list[int] = []
    text_weights: list[float] = []

    for word, weight in texts_and_weights:
        if word == "BREAK":
            # BREAK marker - will be handled during chunking
            text_tokens.append(-1)  # Special marker
            text_weights.append(-1)
            continue

        # Tokenize and discard BOS/EOS tokens
        token_ids = tokenizer(word, truncation=False).input_ids[1:-1]

        text_tokens.extend(token_ids)
        # Expand weight for each token
        text_weights.extend([weight] * len(token_ids))

    return text_tokens, text_weights


def group_tokens_into_chunks(
    token_ids: list[int],
    weights: list[float],
    pad_last_block: bool = True,
) -> tuple[list[list[int]], list[list[float]]]:
    """Group tokens into 77-token chunks with BOS/EOS tokens.

    Each chunk is structured as: [BOS] + 75 tokens + [EOS]
    BREAK markers (-1) force a new chunk boundary.

    Args:
        token_ids: List of token IDs (may contain -1 for BREAK)
        weights: List of per-token weights
        pad_last_block: Whether to pad the last chunk to 77 tokens

    Returns:
        Tuple of (chunked_tokens, chunked_weights) as 2D lists
    """
    new_token_ids: list[list[int]] = []
    new_weights: list[list[float]] = []

    current_tokens: list[int] = []
    current_weights: list[float] = []

    for token_id, weight in zip(token_ids, weights, strict=True):
        # BREAK marker forces new chunk
        if token_id == -1:
            if current_tokens:
                # Pad and finalize current chunk
                padding_len = MAX_TOKENS_PER_CHUNK - len(current_tokens)
                chunk = (
                    [BOS_TOKEN_ID] + current_tokens + [EOS_TOKEN_ID] * padding_len + [EOS_TOKEN_ID]
                )
                chunk_weights = [1.0] + current_weights + [1.0] * padding_len + [1.0]
                new_token_ids.append(chunk)
                new_weights.append(chunk_weights)
                current_tokens = []
                current_weights = []
            continue

        current_tokens.append(token_id)
        current_weights.append(weight)

        # Chunk is full
        if len(current_tokens) >= MAX_TOKENS_PER_CHUNK:
            chunk = [BOS_TOKEN_ID] + current_tokens[:MAX_TOKENS_PER_CHUNK] + [EOS_TOKEN_ID]
            chunk_weights = [1.0] + current_weights[:MAX_TOKENS_PER_CHUNK] + [1.0]
            new_token_ids.append(chunk)
            new_weights.append(chunk_weights)
            current_tokens = current_tokens[MAX_TOKENS_PER_CHUNK:]
            current_weights = current_weights[MAX_TOKENS_PER_CHUNK:]

    # Handle remaining tokens
    if current_tokens:
        if pad_last_block:
            padding_len = MAX_TOKENS_PER_CHUNK - len(current_tokens)
            chunk = [BOS_TOKEN_ID] + current_tokens + [EOS_TOKEN_ID] * padding_len + [EOS_TOKEN_ID]
            chunk_weights = [1.0] + current_weights + [1.0] * padding_len + [1.0]
        else:
            chunk = [BOS_TOKEN_ID] + current_tokens + [EOS_TOKEN_ID]
            chunk_weights = [1.0] + current_weights + [1.0]
        new_token_ids.append(chunk)
        new_weights.append(chunk_weights)

    return new_token_ids, new_weights


def pad_to_same_length(
    tokens_a: list[int],
    weights_a: list[float],
    tokens_b: list[int],
    weights_b: list[float],
) -> tuple[list[int], list[float], list[int], list[float]]:
    """Pad two token/weight lists to the same length.

    Args:
        tokens_a, weights_a: First token/weight pair
        tokens_b, weights_b: Second token/weight pair

    Returns:
        Padded versions of all four lists
    """
    len_a = len(tokens_a)
    len_b = len(tokens_b)

    if len_a > len_b:
        tokens_b = tokens_b + [EOS_TOKEN_ID] * (len_a - len_b)
        weights_b = weights_b + [1.0] * (len_a - len_b)
    elif len_b > len_a:
        tokens_a = tokens_a + [EOS_TOKEN_ID] * (len_b - len_a)
        weights_a = weights_a + [1.0] * (len_b - len_a)

    return tokens_a, weights_a, tokens_b, weights_b


def get_weighted_text_embeddings_sd15(
    pipe: Any,
    prompt: str = "",
    negative_prompt: str = "",
    pad_last_block: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate weighted text embeddings for SD 1.x/2.x pipelines.

    Supports prompts longer than 77 tokens by chunking and concatenating embeddings.

    Args:
        pipe: StableDiffusionPipeline instance
        prompt: Positive prompt with optional weight syntax
        negative_prompt: Negative prompt with optional weight syntax
        pad_last_block: Whether to pad the last chunk

    Returns:
        Tuple of (prompt_embeds, negative_prompt_embeds)
    """
    # Get tokens and weights
    prompt_tokens, prompt_weights = get_tokens_and_weights(pipe.tokenizer, prompt)
    neg_tokens, neg_weights = get_tokens_and_weights(pipe.tokenizer, negative_prompt)

    # Pad to same length
    prompt_tokens, prompt_weights, neg_tokens, neg_weights = pad_to_same_length(
        prompt_tokens, prompt_weights, neg_tokens, neg_weights
    )

    # Group into chunks
    prompt_chunks, prompt_chunk_weights = group_tokens_into_chunks(
        prompt_tokens, prompt_weights, pad_last_block=pad_last_block
    )
    neg_chunks, neg_chunk_weights = group_tokens_into_chunks(
        neg_tokens, neg_weights, pad_last_block=pad_last_block
    )

    # Encode each chunk and apply weights
    embeds = []
    neg_embeds = []

    device = pipe.device if hasattr(pipe, "device") else pipe.text_encoder.device
    dtype = pipe.text_encoder.dtype

    for i in range(len(prompt_chunks)):
        # Positive prompt
        token_tensor = torch.tensor([prompt_chunks[i]], dtype=torch.long, device=device)
        weight_tensor = torch.tensor(prompt_chunk_weights[i], dtype=dtype, device=device)

        token_embedding = pipe.text_encoder(token_tensor)[0].squeeze(0)

        # Apply weights
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                token_embedding[j] = token_embedding[j] * weight_tensor[j]

        embeds.append(token_embedding.unsqueeze(0))

        # Negative prompt
        neg_token_tensor = torch.tensor([neg_chunks[i]], dtype=torch.long, device=device)
        neg_weight_tensor = torch.tensor(neg_chunk_weights[i], dtype=dtype, device=device)

        neg_token_embedding = pipe.text_encoder(neg_token_tensor)[0].squeeze(0)

        for j in range(len(neg_weight_tensor)):
            if neg_weight_tensor[j] != 1.0:
                neg_token_embedding[j] = neg_token_embedding[j] * neg_weight_tensor[j]

        neg_embeds.append(neg_token_embedding.unsqueeze(0))

    # Concatenate all chunks
    prompt_embeds = torch.cat(embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    return prompt_embeds, negative_prompt_embeds


def get_weighted_text_embeddings_sdxl(
    pipe: Any,
    prompt: str = "",
    negative_prompt: str = "",
    pad_last_block: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate weighted text embeddings for SDXL pipelines.

    SDXL uses two text encoders (CLIP-L and CLIP-G) that must be processed
    in sync. The embeddings are concatenated along the feature dimension.

    Args:
        pipe: StableDiffusionXLPipeline instance
        prompt: Positive prompt with optional weight syntax
        negative_prompt: Negative prompt with optional weight syntax
        pad_last_block: Whether to pad the last chunk

    Returns:
        Tuple of (prompt_embeds, negative_prompt_embeds,
                  pooled_prompt_embeds, negative_pooled_prompt_embeds)
    """
    # Tokenizer 1 (CLIP-L)
    prompt_tokens_1, prompt_weights_1 = get_tokens_and_weights(pipe.tokenizer, prompt)
    neg_tokens_1, neg_weights_1 = get_tokens_and_weights(pipe.tokenizer, negative_prompt)

    # Tokenizer 2 (CLIP-G)
    prompt_tokens_2, prompt_weights_2 = get_tokens_and_weights(pipe.tokenizer_2, prompt)
    neg_tokens_2, neg_weights_2 = get_tokens_and_weights(pipe.tokenizer_2, negative_prompt)

    # Pad positive and negative to same length (per tokenizer)
    prompt_tokens_1, prompt_weights_1, neg_tokens_1, neg_weights_1 = pad_to_same_length(
        prompt_tokens_1, prompt_weights_1, neg_tokens_1, neg_weights_1
    )
    prompt_tokens_2, prompt_weights_2, neg_tokens_2, neg_weights_2 = pad_to_same_length(
        prompt_tokens_2, prompt_weights_2, neg_tokens_2, neg_weights_2
    )

    # Group into chunks
    prompt_chunks_1, prompt_chunk_weights_1 = group_tokens_into_chunks(
        prompt_tokens_1, prompt_weights_1, pad_last_block=pad_last_block
    )
    neg_chunks_1, neg_chunk_weights_1 = group_tokens_into_chunks(
        neg_tokens_1, neg_weights_1, pad_last_block=pad_last_block
    )
    prompt_chunks_2, prompt_chunk_weights_2 = group_tokens_into_chunks(
        prompt_tokens_2, prompt_weights_2, pad_last_block=pad_last_block
    )
    neg_chunks_2, neg_chunk_weights_2 = group_tokens_into_chunks(
        neg_tokens_2, neg_weights_2, pad_last_block=pad_last_block
    )

    # Encode and apply weights
    embeds = []
    neg_embeds = []
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None

    device = pipe.device if hasattr(pipe, "device") else pipe.text_encoder.device
    dtype = pipe.text_encoder.dtype

    for i in range(len(prompt_chunks_1)):
        # Positive prompt - encoder 1
        token_tensor_1 = torch.tensor([prompt_chunks_1[i]], dtype=torch.long, device=device)
        weight_tensor = torch.tensor(prompt_chunk_weights_1[i], dtype=dtype, device=device)

        prompt_embeds_1 = pipe.text_encoder(token_tensor_1, output_hidden_states=True)
        # Use penultimate layer for SDXL
        prompt_embeds_1_hidden = prompt_embeds_1.hidden_states[-2]

        # Positive prompt - encoder 2
        token_tensor_2 = torch.tensor([prompt_chunks_2[i]], dtype=torch.long, device=device)

        prompt_embeds_2 = pipe.text_encoder_2(token_tensor_2, output_hidden_states=True)
        prompt_embeds_2_hidden = prompt_embeds_2.hidden_states[-2]

        # Pooled embeddings from encoder 2 (only from first chunk)
        if i == 0:
            pooled_prompt_embeds = prompt_embeds_2[0]

        # Concatenate encoders along feature dimension
        token_embedding = torch.cat(
            [prompt_embeds_1_hidden, prompt_embeds_2_hidden], dim=-1
        ).squeeze(0)

        # Apply weights
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                token_embedding[j] = token_embedding[j] * weight_tensor[j]

        embeds.append(token_embedding.unsqueeze(0))

        # Negative prompt - encoder 1
        neg_token_tensor_1 = torch.tensor([neg_chunks_1[i]], dtype=torch.long, device=device)
        neg_weight_tensor = torch.tensor(neg_chunk_weights_1[i], dtype=dtype, device=device)

        neg_prompt_embeds_1 = pipe.text_encoder(neg_token_tensor_1, output_hidden_states=True)
        neg_prompt_embeds_1_hidden = neg_prompt_embeds_1.hidden_states[-2]

        # Negative prompt - encoder 2
        neg_token_tensor_2 = torch.tensor([neg_chunks_2[i]], dtype=torch.long, device=device)

        neg_prompt_embeds_2 = pipe.text_encoder_2(neg_token_tensor_2, output_hidden_states=True)
        neg_prompt_embeds_2_hidden = neg_prompt_embeds_2.hidden_states[-2]

        # Pooled embeddings from encoder 2 (only from first chunk)
        if i == 0:
            negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        # Concatenate encoders along feature dimension
        neg_token_embedding = torch.cat(
            [neg_prompt_embeds_1_hidden, neg_prompt_embeds_2_hidden], dim=-1
        ).squeeze(0)

        # Apply weights
        for j in range(len(neg_weight_tensor)):
            if neg_weight_tensor[j] != 1.0:
                neg_token_embedding[j] = neg_token_embedding[j] * neg_weight_tensor[j]

        neg_embeds.append(neg_token_embedding.unsqueeze(0))

    # Concatenate all chunks along sequence dimension
    prompt_embeds = torch.cat(embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def is_long_prompt(tokenizer: Any, prompt: str) -> bool:
    """Check if a prompt exceeds the 77-token limit.

    Args:
        tokenizer: CLIP tokenizer
        prompt: Prompt text (weight syntax is stripped for counting)

    Returns:
        True if prompt has more than 75 content tokens
    """
    tokens, _ = get_tokens_and_weights(tokenizer, prompt)
    # Filter out BREAK markers
    tokens = [t for t in tokens if t != -1]
    return len(tokens) > MAX_TOKENS_PER_CHUNK


def needs_long_prompt_handling(pipe: Any, prompt: str, negative_prompt: str | None) -> bool:
    """Check if either prompt needs long prompt handling.

    Args:
        pipe: Pipeline with tokenizer attribute
        prompt: Positive prompt
        negative_prompt: Negative prompt (may be None)

    Returns:
        True if either prompt exceeds token limit
    """
    tokenizer = pipe.tokenizer
    if is_long_prompt(tokenizer, prompt):
        return True
    if negative_prompt and is_long_prompt(tokenizer, negative_prompt):
        return True
    return False
