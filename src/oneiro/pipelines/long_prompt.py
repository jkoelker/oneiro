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
CHUNK_SIZE = 77  # BOS + 75 content tokens + EOS

# T5-XXL embedding dimension
T5_EMBEDDING_DIM = 4096


def _get_execution_device(pipe: Any) -> torch.device | str:
    """Get the execution device for a pipeline, handling CPU offload correctly.

    When using enable_model_cpu_offload(), pipe.device returns "cpu" but actual
    computation happens on GPU. This function returns the correct execution device.

    Args:
        pipe: Diffusers pipeline instance

    Returns:
        The device where tensor operations should happen
    """

    def _is_valid_device(val: Any) -> bool:
        return isinstance(val, (torch.device, str))

    # For CPU offload, _execution_device returns the actual GPU
    if hasattr(pipe, "_execution_device"):
        device = pipe._execution_device
        if _is_valid_device(device):
            return device

    # Standard device detection
    if hasattr(pipe, "device"):
        device = pipe.device
        if _is_valid_device(device):
            return device

    # Fallback to encoder devices
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        if hasattr(pipe.text_encoder, "device"):
            device = pipe.text_encoder.device
            if _is_valid_device(device):
                return device

    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        if hasattr(pipe.text_encoder_2, "device"):
            device = pipe.text_encoder_2.device
            if _is_valid_device(device):
                return device

    # Last resort - check hardware availability
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _create_empty_chunk() -> tuple[list[int], list[float]]:
    """Create an empty 77-token chunk with BOS/EOS structure.

    Returns:
        Tuple of (chunk, weights) where chunk is [BOS] + 76*[EOS]
        and weights is all 1.0s.
    """
    chunk = [BOS_TOKEN_ID] + [EOS_TOKEN_ID] * (MAX_TOKENS_PER_CHUNK + 1)
    weights = [1.0] * CHUNK_SIZE
    return chunk, weights


def _apply_weights_to_embedding(
    embedding: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply per-token weights to embedding tensor in-place.

    Uses vectorized multiplication to efficiently scale each token's
    embedding vector by its corresponding weight.

    Args:
        embedding: Token embeddings of shape (num_tokens, hidden_dim)
        weights: Weight values of shape (num_weights,)

    Returns:
        The modified embedding tensor (modified in-place)
    """
    num_tokens = min(len(weights), embedding.size(0))
    embedding[:num_tokens] = embedding[:num_tokens] * weights[:num_tokens].unsqueeze(-1)
    return embedding


# Regex for parsing A1111-style attention weights
RE_ATTENTION = re.compile(
    r"""
    \\\(|           # escaped (
    \\\)|           # escaped )
    \\\[|           # escaped [
    \\\]|           # escaped ]
    \\\\|           # escaped \
    \\|             # single escape
    \(|             # opening (
    \[|             # opening [
    :([+-]?[.\d]+)\)|  # weight like :1.5)
    \)|             # closing )
    \]|             # closing ]
    [^\\()\[\]:]+|  # regular text
    :               # colon
    """,
    re.X,
)

# BREAK keyword splits prompt into separate chunks
RE_BREAK = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str) -> list[tuple[str, float]]:
    r"""Parse prompt with A1111-style attention weights.

    Supported syntax:
      (abc) - increases attention by 1.1x
      (abc:1.5) - increases attention by 1.5x
      [abc] - decreases attention by dividing by 1.1 (multiplies by ~0.909)
      \( \) \[ \] - literal brackets
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

    # Merge runs of identical weights, but never merge BREAK markers
    # BREAK markers have weight -1 and must remain separate entries
    i = 0
    while i + 1 < len(res):
        # Skip merging if either entry is a BREAK marker
        if res[i][0] == "BREAK" or res[i + 1][0] == "BREAK":
            i += 1
        elif res[i][1] == res[i + 1][1]:
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
        result = tokenizer(word, truncation=False)
        input_ids = result.input_ids if hasattr(result, "input_ids") else []

        # Handle edge cases: empty result or insufficient tokens
        if len(input_ids) < 2:
            # No content tokens (just BOS/EOS or empty), skip this fragment
            continue

        # Strip BOS and EOS tokens (first and last)
        token_ids = input_ids[1:-1]

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

    Each chunk is structured as: [BOS] + content tokens + [EOS padding] + [EOS]
    where content tokens are up to 75 tokens and EOS padding fills the remaining
    space to reach 77 total tokens. For a full chunk: [BOS] + 75 tokens + [EOS].
    For a partial chunk with N tokens (N < 75): [BOS] + N tokens + (75-N) EOS + [EOS].

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
                chunk = [BOS_TOKEN_ID] + current_tokens + [EOS_TOKEN_ID] * (padding_len + 1)
                chunk_weights = [1.0] + current_weights + [1.0] * (padding_len + 1)
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
            chunk = [BOS_TOKEN_ID] + current_tokens + [EOS_TOKEN_ID] * (padding_len + 1)
            chunk_weights = [1.0] + current_weights + [1.0] * (padding_len + 1)
        else:
            chunk = [BOS_TOKEN_ID] + current_tokens + [EOS_TOKEN_ID]
            chunk_weights = [1.0] + current_weights + [1.0]
        new_token_ids.append(chunk)
        new_weights.append(chunk_weights)

    # Handle edge case: if no chunks were produced (e.g., token_ids was empty or only BREAK markers),
    # create an empty chunk to ensure encoders have at least one chunk to process
    if not new_token_ids:
        empty_chunk, empty_weights = _create_empty_chunk()
        new_token_ids.append(empty_chunk)
        new_weights.append(empty_weights)

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


def pad_chunks_to_same_count(
    chunks_a: list[list[int]],
    weights_a: list[list[float]],
    chunks_b: list[list[int]],
    weights_b: list[list[float]],
) -> tuple[list[list[int]], list[list[float]], list[list[int]], list[list[float]]]:
    """Pad two chunk lists to the same number of chunks.

    When positive and negative prompts have different numbers of BREAK markers,
    they will produce different numbers of chunks. This function pads the shorter
    list with empty (EOS-filled) chunks to ensure they can be processed together.

    Args:
        chunks_a, weights_a: First chunk/weight list pair
        chunks_b, weights_b: Second chunk/weight list pair

    Returns:
        Padded versions of all four lists
    """
    len_a = len(chunks_a)
    len_b = len(chunks_b)

    if len_a == len_b:
        return chunks_a, weights_a, chunks_b, weights_b

    # Create an empty chunk (all EOS tokens with weight 1.0)
    # Chunk structure: [BOS] + 75 EOS tokens + [EOS] = 77 tokens
    empty_chunk, empty_weights = _create_empty_chunk()

    if len_a > len_b:
        for _ in range(len_a - len_b):
            chunks_b = chunks_b + [empty_chunk]
            weights_b = weights_b + [empty_weights]
    else:
        for _ in range(len_b - len_a):
            chunks_a = chunks_a + [empty_chunk]
            weights_a = weights_a + [empty_weights]

    return chunks_a, weights_a, chunks_b, weights_b


def _prepare_clip_chunks(
    tokenizer: Any,
    prompt: str,
    negative_prompt: str,
    pad_last_block: bool = True,
) -> tuple[list[list[int]], list[list[float]], list[list[int]], list[list[float]]]:
    """Prepare aligned token chunks for prompt and negative prompt.

    Performs the full tokenization pipeline:
    1. Tokenize prompt and negative with weights
    2. Pad to same length
    3. Group into 77-token chunks
    4. Align chunk counts between prompt and negative

    Args:
        tokenizer: CLIP tokenizer instance
        prompt: Positive prompt with optional weight syntax
        negative_prompt: Negative prompt with optional weight syntax
        pad_last_block: Whether to pad the last chunk to 77 tokens

    Returns:
        Tuple of (prompt_chunks, prompt_weights, neg_chunks, neg_weights)
        where each is a list of lists (one per chunk)
    """
    prompt_tokens, prompt_weights = get_tokens_and_weights(tokenizer, prompt)
    neg_tokens, neg_weights = get_tokens_and_weights(tokenizer, negative_prompt)

    prompt_tokens, prompt_weights, neg_tokens, neg_weights = pad_to_same_length(
        prompt_tokens, prompt_weights, neg_tokens, neg_weights
    )

    prompt_chunks, prompt_chunk_weights = group_tokens_into_chunks(
        prompt_tokens, prompt_weights, pad_last_block=pad_last_block
    )
    neg_chunks, neg_chunk_weights = group_tokens_into_chunks(
        neg_tokens, neg_weights, pad_last_block=pad_last_block
    )

    return pad_chunks_to_same_count(
        prompt_chunks, prompt_chunk_weights, neg_chunks, neg_chunk_weights
    )


def _encode_clip_chunk(
    encoder: Any,
    chunk: list[int],
    weights: list[float],
    device: torch.device | str,
    dtype: torch.dtype,
    use_penultimate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode a single CLIP chunk and apply per-token weights.

    Args:
        encoder: CLIP text encoder
        chunk: Token IDs for one 77-token chunk
        weights: Per-token weights matching chunk length
        device: Target device for tensors
        dtype: Target dtype for weight tensor
        use_penultimate: If True, extract hidden_states[-2] (for SDXL/SD3).
                        If False, use direct output[0] (for SD15).

    Returns:
        Tuple of (weighted_embedding, pooled_output)
        - weighted_embedding: Shape [seq_len, hidden_dim] with weights applied
        - pooled_output: Pooled embedding if use_penultimate, else None
    """
    token_tensor = torch.tensor([chunk], dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weights, dtype=dtype, device=device)

    with torch.no_grad():
        if use_penultimate:
            output = encoder(token_tensor, output_hidden_states=True)
            embedding = output.hidden_states[-2].squeeze(0)
            pooled = output[0]
        else:
            embedding = encoder(token_tensor)[0].squeeze(0)
            pooled = None

    _apply_weights_to_embedding(embedding, weight_tensor)
    return embedding, pooled


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
    prompt_chunks, prompt_weights, neg_chunks, neg_weights = _prepare_clip_chunks(
        pipe.tokenizer, prompt, negative_prompt, pad_last_block
    )

    embeds = []
    neg_embeds = []

    device = _get_execution_device(pipe)
    dtype = (
        pipe.text_encoder.dtype
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None
        else torch.float32
    )

    for i in range(len(prompt_chunks)):
        embedding, _ = _encode_clip_chunk(
            pipe.text_encoder,
            prompt_chunks[i],
            prompt_weights[i],
            device,
            dtype,
            use_penultimate=False,
        )
        embeds.append(embedding.unsqueeze(0))

        neg_embedding, _ = _encode_clip_chunk(
            pipe.text_encoder,
            neg_chunks[i],
            neg_weights[i],
            device,
            dtype,
            use_penultimate=False,
        )
        neg_embeds.append(neg_embedding.unsqueeze(0))

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

        # Tokenize without special tokens for fragments to avoid duplicate EOS
        token_ids = tokenizer(word, truncation=False, add_special_tokens=False).input_ids

        text_tokens.extend(token_ids)
        text_weights.extend([weight] * len(token_ids))

    # Add EOS token at the end (T5's EOS token ID is typically 1)
    # Always add EOS, even if text_tokens is empty (e.g., prompt was only BREAK keywords)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        text_tokens.append(eos_token_id)
        text_weights.append(1.0)

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
    prompt_chunks_1, prompt_weights_1, neg_chunks_1, neg_weights_1 = _prepare_clip_chunks(
        pipe.tokenizer, prompt, negative_prompt, pad_last_block
    )
    prompt_chunks_2, prompt_weights_2, neg_chunks_2, neg_weights_2 = _prepare_clip_chunks(
        pipe.tokenizer_2, prompt, negative_prompt, pad_last_block
    )
    embeds = []
    neg_embeds = []
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None

    device = _get_execution_device(pipe)
    dtype = pipe.text_encoder.dtype

    for i in range(len(prompt_chunks_1)):
        token_tensor_1 = torch.tensor([prompt_chunks_1[i]], dtype=torch.long, device=device)
        weight_tensor = torch.tensor(prompt_weights_1[i], dtype=dtype, device=device)

        prompt_embeds_1 = pipe.text_encoder(token_tensor_1, output_hidden_states=True)
        prompt_embeds_1_hidden = prompt_embeds_1.hidden_states[-2]

        token_tensor_2 = torch.tensor([prompt_chunks_2[i]], dtype=torch.long, device=device)
        prompt_embeds_2 = pipe.text_encoder_2(token_tensor_2, output_hidden_states=True)
        prompt_embeds_2_hidden = prompt_embeds_2.hidden_states[-2]

        if i == 0:
            pooled_prompt_embeds = prompt_embeds_2[0]

        token_embedding = torch.cat(
            [prompt_embeds_1_hidden, prompt_embeds_2_hidden], dim=-1
        ).squeeze(0)
        _apply_weights_to_embedding(token_embedding, weight_tensor)
        embeds.append(token_embedding.unsqueeze(0))

        neg_token_tensor_1 = torch.tensor([neg_chunks_1[i]], dtype=torch.long, device=device)
        neg_weight_tensor = torch.tensor(neg_weights_1[i], dtype=dtype, device=device)

        neg_prompt_embeds_1 = pipe.text_encoder(neg_token_tensor_1, output_hidden_states=True)
        neg_prompt_embeds_1_hidden = neg_prompt_embeds_1.hidden_states[-2]

        neg_token_tensor_2 = torch.tensor([neg_chunks_2[i]], dtype=torch.long, device=device)
        neg_prompt_embeds_2 = pipe.text_encoder_2(neg_token_tensor_2, output_hidden_states=True)
        neg_prompt_embeds_2_hidden = neg_prompt_embeds_2.hidden_states[-2]

        if i == 0:
            negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        neg_token_embedding = torch.cat(
            [neg_prompt_embeds_1_hidden, neg_prompt_embeds_2_hidden], dim=-1
        ).squeeze(0)
        _apply_weights_to_embedding(neg_token_embedding, neg_weight_tensor)
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
    device = _get_execution_device(pipe)

    # Use "empty" placeholder for empty prompts (consistent with get_tokens_and_weights)
    effective_prompt = prompt if prompt else "empty"

    # Get CLIP tokens for pooled embeddings (uses chunking for long prompts)
    clip_tokens, clip_weights = get_tokens_and_weights(pipe.tokenizer, effective_prompt)
    # Note: group_tokens_into_chunks guarantees at least one chunk, and
    # clip_chunk_weights is not used for pooled embeddings
    clip_chunks, _ = group_tokens_into_chunks(clip_tokens, clip_weights, pad_last_block=True)

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
    t5_weight_tensor = torch.tensor(t5_weights, dtype=t5_embeds.dtype, device=device)
    _apply_weights_to_embedding(t5_embeds, t5_weight_tensor)

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
    device = _get_execution_device(pipe)

    prompt_chunks_1, prompt_weights_1, neg_chunks_1, neg_weights_1 = _prepare_clip_chunks(
        pipe.tokenizer, prompt, negative_prompt, pad_last_block
    )
    prompt_chunks_2, prompt_weights_2, neg_chunks_2, neg_weights_2 = _prepare_clip_chunks(
        pipe.tokenizer_2, prompt, negative_prompt, pad_last_block
    )

    prompt_tokens_3, prompt_weights_3 = get_t5_tokens_and_weights(pipe.tokenizer_3, prompt)
    neg_tokens_3, neg_weights_3 = get_t5_tokens_and_weights(pipe.tokenizer_3, negative_prompt)

    # Encode CLIP chunks
    embeds = []
    neg_embeds = []
    pooled_prompt_embeds_1 = None
    pooled_prompt_embeds_2 = None
    negative_pooled_prompt_embeds_1 = None
    negative_pooled_prompt_embeds_2 = None

    # Determine dtype - use text_encoder if available, otherwise use text_encoder_2's dtype
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        dtype = pipe.text_encoder.dtype
    elif hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        dtype = pipe.text_encoder_2.dtype
    else:
        dtype = torch.float32  # Fallback default

    for i in range(len(prompt_chunks_1)):
        # Positive - encoder 1
        token_tensor_1 = torch.tensor([prompt_chunks_1[i]], dtype=torch.long, device=device)
        weight_tensor = torch.tensor(prompt_weights_1[i], dtype=dtype, device=device)

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
        _apply_weights_to_embedding(token_embedding, weight_tensor)

        embeds.append(token_embedding.unsqueeze(0))

        # Negative - encoder 1
        neg_token_tensor_1 = torch.tensor([neg_chunks_1[i]], dtype=torch.long, device=device)
        neg_weight_tensor = torch.tensor(neg_weights_1[i], dtype=dtype, device=device)

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
        _apply_weights_to_embedding(neg_token_embedding, neg_weight_tensor)

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

        # Apply weights
        t5_weight_tensor = torch.tensor(prompt_weights_3, dtype=t5_embeds.dtype, device=device)
        _apply_weights_to_embedding(t5_embeds, t5_weight_tensor)
        t5_embeds = t5_embeds.unsqueeze(0)

        # Negative T5
        neg_t5_token_tensor = torch.tensor([neg_tokens_3], dtype=torch.long, device=device)
        neg_t5_embeds = pipe.text_encoder_3(neg_t5_token_tensor)[0].squeeze(0)

        # Apply weights
        neg_t5_weight_tensor = torch.tensor(neg_weights_3, dtype=neg_t5_embeds.dtype, device=device)
        _apply_weights_to_embedding(neg_t5_embeds, neg_t5_weight_tensor)
        neg_t5_embeds = neg_t5_embeds.unsqueeze(0)
    else:
        # Create zero tensors if T5 not available
        t5_embeds = torch.zeros(
            1, 1, T5_EMBEDDING_DIM, dtype=clip_prompt_embeds.dtype, device=device
        )
        neg_t5_embeds = torch.zeros(
            1, 1, T5_EMBEDDING_DIM, dtype=clip_prompt_embeds.dtype, device=device
        )

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
