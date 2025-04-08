import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing_extensions import Optional, Union, Tuple

from core.bark.constants import *
from core.model.bark import GPT
from core.data_model.bark import BarkPrompt
from core.bark.custom_context import inference_mode

from core.memory import model_manager, ModelEnum, env

# number of coarse tokens per one semantic token for one second
num_coarse_per_semantic = (COARSE_RATE_HZ / SEMANTIC_RATE_HZ) * N_COARSE_CODEBOOKS


def generate_coarse_tokens_from_semantic(
    semantic_tokens: torch.Tensor,
    history_prompt: Union[BarkPrompt, None] = None,
    generate_coarse_temperature: Union[float, None] = 0.6,
    coarse_top_k: Union[int, None] = None,
    coarse_top_p: Union[float, None] = None,
    silent: bool = False,
    max_coarse_history: int = 630,
    sliding_window_length: int = 60,
    use_kv_caching: bool = True,
    use_small_model: bool = False,
    **kwargs,
) -> torch.Tensor:
    # Validate inputs
    _validate_semantic_tokens(semantic_tokens)
    _validate_history_prompt(history_prompt)

    assert (
        60 <= max_coarse_history <= 630
    ), "max_coarse_history must be between 60 and 630"
    assert (
        max_coarse_history + sliding_window_length <= 1024 - 256
    ), "Context exceeds model limit"

    # align the number of semantic history token with the given max_coarse_history
    max_semantic_history = int(max_coarse_history / num_coarse_per_semantic)

    # align the length of the provided semantic and coarse history
    semantic_history, coarse_history = _process_history_prompt(
        history_prompt, max_semantic_history, num_coarse_per_semantic
    )

    # Load coarse model
    coarse_model_info = (
        ModelEnum.BARK_COARSE_SMALL.value
        if use_small_model
        else ModelEnum.BARK_COARSE.value
    )
    model_wrapper = model_manager.get_model(coarse_model_info)
    model: GPT = model_wrapper.model
    assert isinstance(model, GPT), "unexpected model type"

    # total_steps is the number of coarse tokens the model need to predict
    total_steps = int(
        np.floor(semantic_tokens.size(1) * num_coarse_per_semantic / N_COARSE_CODEBOOKS)
        * N_COARSE_CODEBOOKS
    )
    assert (
        total_steps > 0 and total_steps % N_COARSE_CODEBOOKS == 0
    ), "Invalid step count"

    batch, T = semantic_tokens.size()
    # expand the semantic history at the batch dimension to match with the semantic_tokens tensor's batch size
    # for the concatenation
    semantic_history = semantic_history[None].expand((batch, -1))
    full_semantic = torch.hstack([semantic_history, semantic_tokens]).to(torch.int32)
    base_semantic_index = semantic_history.size(1)

    # Generate coarse tokens
    with inference_mode():
        generated_coarse = _generate_coarse_with_sliding_window(
            model,
            full_semantic,
            coarse_history,
            total_steps,
            base_semantic_index,
            max_semantic_history,
            num_coarse_per_semantic,
            generate_coarse_temperature,
            coarse_top_k,
            coarse_top_p,
            silent,
            max_coarse_history,
            sliding_window_length,
            use_kv_caching,
        )

    # remove the history prompt from the generated tokens
    generated_coarse = generated_coarse[:, coarse_history.size(0) :]
    assert generated_coarse.size(1) == total_steps, "Generated length mismatch"

    # Reshape and adjust coarse codes
    B, L = generated_coarse.shape
    # Broadcasting subtracts from all elements
    coarse_output = (
        generated_coarse.reshape(B, -1, N_COARSE_CODEBOOKS).transpose(1, 2)
        - SEMANTIC_VOCAB_SIZE
    )

    for codebook_idx in range(1, N_COARSE_CODEBOOKS):
        coarse_output[:, codebook_idx, :] -= codebook_idx * CODEBOOK_SIZE

    return coarse_output


def _validate_semantic_tokens(semantic_tokens: torch.Tensor) -> None:
    """
    Validate the input semantic tokens tensor.

    Args:
        semantic_tokens: Tensor of semantic tokens (1D).

    Raises:
        AssertionError: If the tensor does not meet expected criteria.
    """
    assert isinstance(
        semantic_tokens, torch.Tensor
    ), "Semantic tokens must be a torch.Tensor"
    assert semantic_tokens.dim() == 2, "Semantic tokens must be 2D"
    assert semantic_tokens.size(1) > 0, "Semantic tokens tensor cannot be empty"
    assert semantic_tokens.min() >= 0, "Semantic tokens must be non-negative"
    assert (
        semantic_tokens.max() <= SEMANTIC_VOCAB_SIZE
    ), "Semantic tokens exceed vocab size"


def _validate_history_prompt(history_prompt: Union[BarkPrompt, None]) -> None:
    """
    Validate the history prompt if provided.

    Args:
        history_prompt: BarkPrompt object or None.

    Raises:
        AssertionError: If the prompt does not meet expected criteria.
    """
    if history_prompt is None:
        return

    assert isinstance(
        history_prompt, BarkPrompt
    ), "History prompt must be a BarkPrompt object"
    semantic = history_prompt.semantic_prompt
    coarse = history_prompt.coarse_prompt

    assert (
        isinstance(semantic, torch.Tensor) and semantic.dim() == 1
    ), "Semantic prompt must be 1D tensor"
    assert (
        semantic.size(0) > 0
        and semantic.min() >= 0
        and semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
    )
    assert (
        isinstance(coarse, torch.Tensor) and coarse.dim() == 2
    ), "Coarse prompt must be 2D tensor"
    assert (
        coarse.shape[0] == N_COARSE_CODEBOOKS
    ), "Coarse prompt must have correct number of codebooks"
    assert coarse.min() >= 0 and coarse.max() <= CODEBOOK_SIZE - 1


def _process_history_prompt(
    history_prompt: Union[BarkPrompt, None],
    max_semantic_history: int,
    coarse_to_semantic_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process the history prompt into semantic and coarse history tensors.
    Trim on the left (keep the right most tokens)
    Args:
        history_prompt: BarkPrompt object or None.
        max_semantic_history: Maximum number of semantic history tokens.
        coarse_to_semantic_ratio: Ratio of coarse to semantic token rates.

    Returns:
        Tuple[semantic_history, coarse_history]: Processed history tensors.
    """
    if history_prompt is None:
        return torch.tensor(
            [], dtype=torch.int32, device=torch.device(env.DEVICE)
        ), torch.tensor([], dtype=torch.int32, device=torch.device(env.DEVICE))

    semantic_history = history_prompt.semantic_prompt
    coarse_history = history_prompt.coarse_prompt

    # Add offset then "ravel("F")" flatten
    coarse_history = _add_codebook_offset(coarse_history, CODEBOOK_SIZE)
    coarse_history_flat = coarse_history.T.flatten() + SEMANTIC_VOCAB_SIZE

    # Trim histories to fit max length
    n_semantic_hist = min(
        max_semantic_history,
        semantic_history.size(0) - semantic_history.size(0) % 2,  # Ensure even length
        int(coarse_history_flat.size(0) // coarse_to_semantic_ratio),
    )
    n_coarse_hist = int(round(n_semantic_hist * coarse_to_semantic_ratio))

    semantic_history = semantic_history[-n_semantic_hist:].to(torch.int32)
    coarse_history_flat = coarse_history_flat[-n_coarse_hist:].to(torch.int32)
    coarse_history_flat = coarse_history_flat[:-2]  # Original time alignment hack

    return semantic_history, coarse_history_flat


def _add_codebook_offset(x: torch.Tensor, offset: int) -> torch.Tensor:
    """
    x shape (n_codebook, T)
    n_codebook start from 0 to n, from the second codebook row on we add offset * row_num
    """
    for n in range(1, x.shape[0]):
        x[n, :] += offset * n
    return x


def _sample_coarse_token(
    logits: torch.Tensor,
    temperature: Union[float, None],
    top_k: Optional[int],
    top_p: Optional[float],
    logit_start_idx: int,
) -> torch.Tensor:
    """
    Sample a coarse token from model logits with filtering.

    Args:
        logits: Model output logits (shape [batch, seq, vocab]).
        temperature: Sampling temperature for randomness.
        top_k: Number of top logits to consider, if specified.
        top_p: Nucleus sampling threshold, if specified.
        logit_start_idx: Starting index for coarse token logits.

    Returns:
        torch.Tensor: Sampled token with offset applied (shape [1]).
    """
    relevant_logits = logits[:, 0, logit_start_idx : logit_start_idx + CODEBOOK_SIZE]

    if temperature is None:
        probs = F.softmax(relevant_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True).to(torch.int32)
    else:
        if top_p is not None:  # this branch is untested
            # Optimize with NumPy for top-p filtering,
            original_device = relevant_logits.device
            logits_np = relevant_logits.detach().cpu().numpy().astype(np.float32)
            sorted_indices = np.argsort(logits_np)[::-1]
            sorted_logits = logits_np[sorted_indices]
            cumulative_probs = np.cumsum(
                F.softmax(torch.from_numpy(sorted_logits), dim=-1).numpy()
            )
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[1:] = indices_to_remove[:-1].copy()
            indices_to_remove[0] = False
            logits_np[sorted_indices[indices_to_remove]] = -np.inf
            relevant_logits = torch.from_numpy(logits_np).to(original_device)

        if top_k is not None:
            top_values, _ = torch.topk(
                relevant_logits, min(top_k, relevant_logits.size(-1))
            )
            relevant_logits[relevant_logits < top_values[:, [-1]]] = -float("Inf")

        probs = F.softmax(relevant_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).to(torch.int32)
    return next_token + logit_start_idx


def _generate_coarse_with_sliding_window(
    model: GPT,
    full_semantic: torch.Tensor,
    coarse_history: torch.Tensor,
    total_steps: int,
    base_semantic_index: int,
    max_semantic_history: int,
    coarse_per_semantic: float,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    silent: bool,
    max_coarse_history: int,
    sliding_window_length: int,
    use_kv_caching: bool,
) -> torch.Tensor:
    """
    Generate coarse tokens using a sliding window approach.

    Args:
        model: GPT model for coarse token generation.
        full_semantic: 2D tensor of Concatenated semantic history and input tokens.
        coarse_history: 1D tensor, Initial coarse history tokens.
        total_steps: Total number of coarse tokens to generate.
        base_semantic_index: Start index of input semantic tokens.
        max_semantic_history: Maximum semantic history length.
        coarse_per_semantic: Coarse-to-semantic token ratio.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        top_p: Top-p filtering parameter.
        silent: Suppresses progress bar if True.
        max_coarse_history: Maximum coarse history length.
        sliding_window_length: Tokens per window.
        use_kv_caching: Enables KV caching.

    Returns:
        torch.Tensor: Generated coarse tokens (1D).
    """
    device = next(model.parameters()).device
    semantic_tensor = full_semantic.to(device)  # Add batch dimension
    coarse_tensor = (
        coarse_history[None].expand((semantic_tensor.shape[0], -1)).to(device)
    )

    window_count = int(np.ceil(total_steps / sliding_window_length))
    progress_bar = tqdm(
        total=window_count, disable=silent, desc="Generating coarse tokens"
    )
    step_counter = 0  # equivalent to the number of coarse tokens generated so far

    for _ in range(window_count):
        current_semantic_idx = base_semantic_index + int(
            round(step_counter / coarse_per_semantic)
        )

        window_start = max(0, current_semantic_idx - max_semantic_history)
        semantic_window = semantic_tensor[:, window_start : window_start + 256]
        semantic_window = F.pad(
            semantic_window,
            (0, 256 - semantic_window.shape[-1]),
            "constant",
            COARSE_SEMANTIC_PAD_TOKEN,
        )

        input_tensor = torch.hstack(
            [
                semantic_window,
                torch.tensor([COARSE_INFER_TOKEN], device=device)[None].expand(
                    (semantic_window.shape[0], -1)
                ),
                coarse_tensor[:, -max_coarse_history:],
            ]
        )

        kv_cache = None
        for _ in range(sliding_window_length):
            if step_counter >= total_steps:
                break

            is_first_codebook = step_counter % N_COARSE_CODEBOOKS == 0
            logit_start_idx = (
                SEMANTIC_VOCAB_SIZE + (1 - int(is_first_codebook)) * CODEBOOK_SIZE
            )

            model_input = (
                input_tensor[:, [-1]]
                if use_kv_caching and kv_cache is not None
                else input_tensor
            )
            logits, kv_cache = model(
                model_input, use_cache=use_kv_caching, past_kv=kv_cache
            )
            next_token = _sample_coarse_token(
                logits,
                temperature,
                top_k,
                top_p,
                logit_start_idx,
            )

            coarse_tensor = torch.cat((coarse_tensor, next_token), dim=1)
            input_tensor = torch.cat((input_tensor, next_token), dim=1)

            step_counter += 1
            del logits, next_token

        del input_tensor
        progress_bar.update(1)

    progress_bar.close()
    return coarse_tensor
