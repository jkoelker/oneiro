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


def get_t5_tokens_and_weights(
    tokenizer: Any,
    prompt: str,
) -> tuple[list[int], list[float]]:
    """Tokenize prompt for T5 encoder and extract per-token weights.

    T5 uses different tokenization than CLIP - no BOS/EOS stripping needed,
    and tokens are added with add_special_tokens=True.

    Args:
        tokenizer: T5 tokenizer
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
            # BREAK not applicable for T5 (no chunking), skip
            continue

        # Tokenize with special tokens for T5
        token_ids = tokenizer(word, truncation=False, add_special_tokens=True).input_ids

        text_tokens.extend(token_ids)
        text_weights.extend([weight] * len(token_ids))

    return text_tokens, text_weights


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


def get_weighted_text_embeddings_flux(
    pipe: Any,
    prompt: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate weighted text embeddings for Flux pipelines.

    Flux uses two text encoders:
    - CLIP ViT-L/14 (text_encoder) for pooled embeddings
    - T5-v1.1-XXL (text_encoder_2) for main prompt embeddings

    Weights are applied to T5 embeddings since that's where the semantic
    content is primarily encoded. CLIP pooled embeddings are averaged
    across chunks for long prompts.

    Args:
        pipe: FluxPipeline instance
        prompt: Prompt with optional weight syntax

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds)
        - prompt_embeds: T5 embeddings with weights applied
        - pooled_prompt_embeds: Averaged CLIP pooled embeddings
    """
    device = pipe.device if hasattr(pipe, "device") else "cuda"
    if not str(device).startswith("cuda"):
        device = "cuda" if torch.cuda.is_available() else device

    # Use "empty" placeholder for empty prompts (consistent with get_tokens_and_weights)
    effective_prompt = prompt if prompt else "empty"

    # Get CLIP tokens for pooled embeddings (uses chunking for long prompts)
    clip_tokens, clip_weights = get_tokens_and_weights(pipe.tokenizer, effective_prompt)
    clip_chunks, clip_chunk_weights = group_tokens_into_chunks(
        clip_tokens, clip_weights, pad_last_block=True
    )

    # Get T5 tokens for main embeddings (no chunking needed, T5 handles long sequences)
    t5_tokens, t5_weights = get_t5_tokens_and_weights(pipe.tokenizer_2, effective_prompt)

    # Generate CLIP pooled embeddings (average across chunks)
    pooled_embeds_list = []
    for chunk in clip_chunks:
        token_tensor = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            clip_output = pipe.text_encoder(token_tensor, output_hidden_states=False)
        pooled_embeds_list.append(clip_output.pooler_output.squeeze(0))

    # Average pooled embeddings across chunks
    pooled_prompt_embeds = torch.stack(pooled_embeds_list, dim=0).mean(dim=0, keepdim=True)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=pipe.text_encoder.dtype, device=device)

    # Generate T5 embeddings with weights
    t5_token_tensor = torch.tensor([t5_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        t5_output = pipe.text_encoder_2(t5_token_tensor)
    t5_embeds = t5_output[0].squeeze(0)

    # Apply weights to T5 embeddings
    for i, weight in enumerate(t5_weights):
        if weight != 1.0:
            t5_embeds[i] = t5_embeds[i] * weight

    prompt_embeds = t5_embeds.unsqueeze(0)
    prompt_embeds = prompt_embeds.to(dtype=pipe.text_encoder_2.dtype, device=device)

    return prompt_embeds, pooled_prompt_embeds


def get_weighted_text_embeddings_sd3(
    pipe: Any,
    prompt: str = "",
    negative_prompt: str = "",
    pad_last_block: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate weighted text embeddings for SD3 pipelines.

    SD3 uses three text encoders:
    - CLIP-L (text_encoder) for first set of embeddings
    - CLIP-G (text_encoder_2) for second set of embeddings + pooled
    - T5-XXL (text_encoder_3) for additional semantic embeddings

    The CLIP embeddings are concatenated along the feature dimension,
    then padded and concatenated with T5 embeddings along the sequence dimension.

    Args:
        pipe: StableDiffusion3Pipeline instance
        prompt: Positive prompt with optional weight syntax
        negative_prompt: Negative prompt with optional weight syntax
        pad_last_block: Whether to pad the last CLIP chunk

    Returns:
        Tuple of (prompt_embeds, negative_prompt_embeds,
                  pooled_prompt_embeds, negative_pooled_prompt_embeds)
    """
    device = pipe.device if hasattr(pipe, "device") else pipe.text_encoder.device

    # Tokenizer 1 (CLIP-L)
    prompt_tokens_1, prompt_weights_1 = get_tokens_and_weights(pipe.tokenizer, prompt)
    neg_tokens_1, neg_weights_1 = get_tokens_and_weights(pipe.tokenizer, negative_prompt)

    # Tokenizer 2 (CLIP-G)
    prompt_tokens_2, prompt_weights_2 = get_tokens_and_weights(pipe.tokenizer_2, prompt)
    neg_tokens_2, neg_weights_2 = get_tokens_and_weights(pipe.tokenizer_2, negative_prompt)

    # Tokenizer 3 (T5)
    prompt_tokens_3, prompt_weights_3 = get_t5_tokens_and_weights(pipe.tokenizer_3, prompt)
    neg_tokens_3, neg_weights_3 = get_t5_tokens_and_weights(pipe.tokenizer_3, negative_prompt)

    # Pad CLIP tokens to same length
    prompt_tokens_1, prompt_weights_1, neg_tokens_1, neg_weights_1 = pad_to_same_length(
        prompt_tokens_1, prompt_weights_1, neg_tokens_1, neg_weights_1
    )
    prompt_tokens_2, prompt_weights_2, neg_tokens_2, neg_weights_2 = pad_to_same_length(
        prompt_tokens_2, prompt_weights_2, neg_tokens_2, neg_weights_2
    )

    # Group CLIP tokens into chunks
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

    # Encode CLIP chunks
    embeds = []
    neg_embeds = []
    pooled_prompt_embeds_1 = None
    pooled_prompt_embeds_2 = None
    negative_pooled_prompt_embeds_1 = None
    negative_pooled_prompt_embeds_2 = None

    dtype = pipe.text_encoder.dtype

    for i in range(len(prompt_chunks_1)):
        # Positive - encoder 1
        token_tensor_1 = torch.tensor([prompt_chunks_1[i]], dtype=torch.long, device=device)
        weight_tensor = torch.tensor(prompt_chunk_weights_1[i], dtype=dtype, device=device)

        prompt_embeds_1 = pipe.text_encoder(token_tensor_1, output_hidden_states=True)
        prompt_embeds_1_hidden = prompt_embeds_1.hidden_states[-2]
        if i == 0:
            pooled_prompt_embeds_1 = prompt_embeds_1[0]

        # Positive - encoder 2
        token_tensor_2 = torch.tensor([prompt_chunks_2[i]], dtype=torch.long, device=device)

        prompt_embeds_2 = pipe.text_encoder_2(token_tensor_2, output_hidden_states=True)
        prompt_embeds_2_hidden = prompt_embeds_2.hidden_states[-2]
        if i == 0:
            pooled_prompt_embeds_2 = prompt_embeds_2[0]

        # Concatenate CLIP embeddings
        token_embedding = torch.cat(
            [prompt_embeds_1_hidden, prompt_embeds_2_hidden], dim=-1
        ).squeeze(0)

        # Apply weights
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                token_embedding[j] = token_embedding[j] * weight_tensor[j]

        embeds.append(token_embedding.unsqueeze(0))

        # Negative - encoder 1
        neg_token_tensor_1 = torch.tensor([neg_chunks_1[i]], dtype=torch.long, device=device)
        neg_weight_tensor = torch.tensor(neg_chunk_weights_1[i], dtype=dtype, device=device)

        neg_prompt_embeds_1 = pipe.text_encoder(neg_token_tensor_1, output_hidden_states=True)
        neg_prompt_embeds_1_hidden = neg_prompt_embeds_1.hidden_states[-2]
        if i == 0:
            negative_pooled_prompt_embeds_1 = neg_prompt_embeds_1[0]

        # Negative - encoder 2
        neg_token_tensor_2 = torch.tensor([neg_chunks_2[i]], dtype=torch.long, device=device)

        neg_prompt_embeds_2 = pipe.text_encoder_2(neg_token_tensor_2, output_hidden_states=True)
        neg_prompt_embeds_2_hidden = neg_prompt_embeds_2.hidden_states[-2]
        if i == 0:
            negative_pooled_prompt_embeds_2 = neg_prompt_embeds_2[0]

        # Concatenate CLIP embeddings
        neg_token_embedding = torch.cat(
            [neg_prompt_embeds_1_hidden, neg_prompt_embeds_2_hidden], dim=-1
        ).squeeze(0)

        # Apply weights
        for j in range(len(neg_weight_tensor)):
            if neg_weight_tensor[j] != 1.0:
                neg_token_embedding[j] = neg_token_embedding[j] * neg_weight_tensor[j]

        neg_embeds.append(neg_token_embedding.unsqueeze(0))

    clip_prompt_embeds = torch.cat(embeds, dim=1)
    clip_negative_embeds = torch.cat(neg_embeds, dim=1)

    # Combine pooled embeddings
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
    negative_pooled_prompt_embeds = torch.cat(
        [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1
    )

    # Generate T5 embeddings if text_encoder_3 is available
    if pipe.text_encoder_3 is not None:
        # Positive T5
        t5_token_tensor = torch.tensor([prompt_tokens_3], dtype=torch.long, device=device)
        t5_embeds = pipe.text_encoder_3(t5_token_tensor)[0].squeeze(0)

        for i, weight in enumerate(prompt_weights_3):
            if weight != 1.0:
                t5_embeds[i] = t5_embeds[i] * weight
        t5_embeds = t5_embeds.unsqueeze(0)

        # Negative T5
        neg_t5_token_tensor = torch.tensor([neg_tokens_3], dtype=torch.long, device=device)
        neg_t5_embeds = pipe.text_encoder_3(neg_t5_token_tensor)[0].squeeze(0)

        for i, weight in enumerate(neg_weights_3):
            if weight != 1.0:
                neg_t5_embeds[i] = neg_t5_embeds[i] * weight
        neg_t5_embeds = neg_t5_embeds.unsqueeze(0)
    else:
        # Create zero tensors if T5 not available
        t5_embeds = torch.zeros(1, 1, 4096, dtype=clip_prompt_embeds.dtype, device=device)
        neg_t5_embeds = torch.zeros(1, 1, 4096, dtype=clip_prompt_embeds.dtype, device=device)

    # Pad CLIP embeddings to match T5 dimension and concatenate
    clip_prompt_embeds_padded = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds_padded, t5_embeds], dim=-2)

    clip_negative_embeds_padded = torch.nn.functional.pad(
        clip_negative_embeds, (0, neg_t5_embeds.shape[-1] - clip_negative_embeds.shape[-1])
    )
    negative_prompt_embeds = torch.cat([clip_negative_embeds_padded, neg_t5_embeds], dim=-2)

    # Pad to same sequence length if needed
    size_diff = negative_prompt_embeds.size(1) - prompt_embeds.size(1)
    if size_diff > 0:
        prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, size_diff))
    elif size_diff < 0:
        negative_prompt_embeds = torch.nn.functional.pad(
            negative_prompt_embeds, (0, 0, 0, -size_diff)
        )

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
