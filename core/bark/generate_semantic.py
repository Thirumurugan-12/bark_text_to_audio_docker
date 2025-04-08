from typing import List, Tuple, Optional, Union
import re
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from transformers import BertTokenizer

from core.memory import model_manager, ModelEnum, env
from core.bark.custom_context import inference_mode
from core.bark.constants import *
from core.model import GPT

SEMANTIC_EOS_TOKEN = 10_000


def generate_semantic_tokens_from_text(
    texts: List[str],
    semantic_prompt: Union[torch.Tensor, None] = None,
    temperature: Union[float, None] = 0.7,
    semantic_top_k: Union[int, None] = None,
    semantic_top_p: Union[int, None] = None,
    min_eos_p: float = 0.2,
    max_gen_duration_second: Union[float, None] = None,
    allow_early_stop: bool = True,
    use_kv_caching: bool = True,
    use_small_model: bool = True,
    silent: Union[bool, None] = False,
    max_token_ids_per_sentence: int = 256,
    **kwargs,
) -> torch.Tensor:
    # trim white spaces and replace redundant white space characters
    texts = _preprocess_texts(texts)
    assert all([len(text) > 0 for text in texts]), f"invalid input text {texts}"

    if semantic_prompt is None:
        semantic_prompt = torch.tensor([])
    else:
        assert isinstance(
            semantic_prompt, torch.Tensor
        ), f"expecting semantic_prompt of type torch.Tensor, received {type(semantic_prompt)}"
        assert semantic_prompt.dim() == 1, "expect 1D tensor as semantic_prompt"

    # load the GPT-style model that generate semantic token from text
    # and the BERT tokenizer to memory
    text_model_info = (
        ModelEnum.BARK_TEXT_SMALL.value
        if use_small_model
        else ModelEnum.BARK_TEXT.value
    )

    text_model = model_manager.get_model(text_model_info)
    assert text_model.model is not None, "text model is None"
    assert text_model.preprocessor is not None, "tokenizer for the text model is None"

    assert isinstance(
        text_model.model, GPT
    ), f"expecting model of type GPT, got {type(text_model.model)}"

    assert isinstance(
        text_model.preprocessor, BertTokenizer
    ), f"expecting preprocessor of type BertTokenizer, got {type(text_model.preprocessor)}"

    model: GPT = text_model.model
    tokenizer: BertTokenizer = text_model.preprocessor
    device = next(model.parameters()).device

    # tokenize the given text using the BERT tokenizer
    token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    # for each token_ids of each sentence, append an encoding offset token
    token_ids = [np.array(sentence) + TEXT_ENCODING_OFFSET for sentence in token_ids]

    # encoded_text's length must has length 256 as from the original implementation
    # pad to the right if the token_ids of the sentence is shorter, trim on the right if it is longer than 256 tokens
    token_ids = [
        trim_or_pad_array(sentence, TEXT_PAD_TOKEN, max_token_ids_per_sentence)
        for sentence in token_ids
    ]

    token_ids_tensor = torch.vstack(token_ids).to(dtype=torch.int32, device=device)

    # when the token_ids list has one element (batch size = 1), the above cat operation created a 1D tensor
    # we need to check and make it 2D
    if len(token_ids_tensor.shape) == 1:
        token_ids_tensor = token_ids_tensor.unsqueeze(0)
    # semantic prompt also need to be an array of 256 discrete tokens
    semantic_prompt = trim_or_pad_array(semantic_prompt, SEMANTIC_PAD_TOKEN, 256)

    # need to replicate the semantic_prompt array to match the shape of the token_ids for concatenation
    semantic_prompt = (
        semantic_prompt.unsqueeze(0).expand((token_ids_tensor.shape[0], -1)).to(device)
    )

    # final input is the concatenation of the token_ids and the semantic tokens array
    input_tensor = torch.cat(
        [
            token_ids_tensor,  # shape (batch_size, T)
            semantic_prompt,
            torch.tensor([SEMANTIC_INFER_TOKEN], device=device)
            .unsqueeze(0)
            .expand((token_ids_tensor.shape[0], -1)),
        ],
        dim=1,
    ).to(torch.int64)

    # 256 token_ids, 256 prompt tokens, 1 semantic_infer token as the last column
    assert (
        input_tensor.shape[1] == 256 + 256 + 1
    ), f"expecting tensor shape [batch, 513], received {input_tensor.shape}"

    with inference_mode():
        output: torch.Tensor = _generate_semantic(
            model=model,
            x=input_tensor,
            temperature=temperature,
            top_k=semantic_top_k,
            top_p=semantic_top_p,
            min_eos_p=min_eos_p,
            max_gen_duration_s=max_gen_duration_second,
            allow_early_stop=allow_early_stop,
            use_kv_caching=use_kv_caching,
            silent=silent,
        )

        validate_semantic_token_output(output)
    return output


def _generate_semantic(
    model: GPT,
    x: torch.Tensor,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_eos_p: float = 0.2,
    max_gen_duration_s: Optional[float] = None,
    allow_early_stop: bool = True,
    use_kv_caching: bool = False,
    silent: bool = False,
) -> torch.Tensor:
    # Maximum number of tokens to generate
    max_steps = 2048

    # Initialize progress bar for user feedback (custom due to unpredictable stopping)
    progress_bar = tqdm(
        total=max_steps, disable=silent, desc="Generating semantic tokens"
    )
    last_progress = 0

    # Key-value cache for attention optimization
    kv_cache = None

    # Autoregressive generation loop
    for step in range(max_steps):
        # Determine input based on KV caching
        if use_kv_caching and kv_cache is not None:
            # Use only the last token with cached attention states
            x_input = x[:, [-1]]  # Shape [1, 1]
        else:
            # Use full sequence (recomputes attention each time)
            x_input = x  # Shape [1, seq_len]

        # Forward pass through the model
        logits, kv_cache = model(
            x_input,
            merge_context=True,  # Merges text and semantic history context
            past_kv=kv_cache,  # Previous attention states
            use_cache=use_kv_caching,  # Enables caching if requested
        )

        # Sample the next token and check for early stopping
        next_token, should_stop = _sample_next_token(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            semantic_eos_token=SEMANTIC_EOS_TOKEN,
            allow_early_stop=allow_early_stop,
            min_eos_p=min_eos_p,
        )

        # Check stopping conditions
        # only stop if all generations in the batch reached the stopping condition
        if torch.all(should_stop):
            progress_bar.update(step - last_progress + 1)
            break

        if step == max_steps - 1:
            progress_bar.update()
            break

        # Append the new token to the sequence
        x = torch.cat((x, next_token), dim=1)

        # Update duration and progress
        # total_duration_s += duration_per_step
        if step > last_progress:
            progress_bar.update(step - last_progress)
            last_progress = step

        # Clean up tensors to manage memory
        del logits, next_token

    # Finalize progress bar
    progress_bar.total = step + 1
    progress_bar.close()

    # Extract generated tokens (skip initial 513 context tokens)
    output = x[:, 256 + 256 + 1 :].detach()

    return output


def _sample_next_token(
    logits: torch.Tensor,  # what is the shape of logits?
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    semantic_eos_token: int,
    allow_early_stop: bool,
    min_eos_p: Optional[float],
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Sample the next token from logits with optional top-k, top-p filtering and early stopping.

    Args:
        logits: Tensor of shape [batch, seq, vocab_size] containing model predictions.
        temperature: Controls randomness of sampling (lower = more deterministic).
        top_k: If set, keeps only the top-k logits.
        top_p: If set, applies nucleus (top-p) filtering.
        vocab_size: Size of the semantic vocabulary (e.g., SEMANTIC_VOCAB_SIZE).
        allow_early_stop: Whether to check for EOS token or probability threshold.
        min_eos_p: Minimum probability for EOS to trigger early stop.
        eos_token: Token ID representing end-of-sequence.

    Returns:
        Tuple[next_token, should_stop]:
            - next_token: Sampled token (shape [1]).
            - should_stop: Whether to stop generation (EOS detected).
    """
    # Extract logits for the last position in the sequence
    relevant_logits = logits[:, -1, :semantic_eos_token]

    # Append EOS logit if early stopping is allowed
    if allow_early_stop:
        eos_logit = logits[:, -1, [semantic_eos_token]]
        relevant_logits = torch.hstack((relevant_logits, eos_logit))

    # select the token with the highest probability
    if temperature is None:
        # next_token shape (B, 1)
        probs = F.softmax(relevant_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        # when the model predict a 206 token_id, it continue to predict that same token_id with argmax
        # we will intentionally avoid that token_id here
        if torch.any(next_token == 206):
            next_token = anything_but(probs, 206)

    # do some maneuvers to introduce diversity in the sampling of the next token
    else:
        # Apply top-p (nucleus) filtering for diversity
        if top_p is not None:  # this if branch is untested
            # Convert to NumPy for faster sorting (optimization from original)
            original_device = relevant_logits.device
            logits_np = relevant_logits.detach().cpu().type(torch.float32).numpy()
            sorted_indices = np.argsort(logits_np)[::-1]  # Descending order
            sorted_logits = logits_np[sorted_indices]
            cumulative_probs = np.cumsum(
                F.softmax(torch.from_numpy(sorted_logits), dim=-1).numpy()
            )
            indices_to_remove = cumulative_probs > top_p
            # Shift to keep at least one
            indices_to_remove[1:] = indices_to_remove[:-1].copy()
            indices_to_remove[0] = False  # Ensure top token stays
            logits_np[sorted_indices[indices_to_remove]] = -np.inf
            relevant_logits = torch.from_numpy(logits_np).to(original_device)

        # Apply top-k filtering for diversity
        if top_k is not None:
            top_values, _ = torch.topk(
                relevant_logits, min(top_k, relevant_logits.size(-1))
            )
            # compare the whole logit tensor to its k_th largest value, batch wise
            relevant_logits[relevant_logits < top_values[:, [-1]]] = -float("Inf")

        # Compute probabilities with temperature scaling
        probs = F.softmax(relevant_logits / temperature, dim=-1)

        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1).to(torch.int32)

    # Check for early stopping conditions for each sequence in the batch
    if allow_early_stop:
        # EOS token is vocab_size when appended
        is_eos_token = (next_token == semantic_eos_token).flatten()
        eos_prob_high = min_eos_p is not None and probs[:, -1] >= min_eos_p
        should_stop = torch.logical_or(is_eos_token, eos_prob_high)

    # when batch dimension is 1, next_token is a 1D array, need to make it 2D
    if len(next_token.shape) == 1:
        next_token = next_token.unsqueeze(0)
    return next_token, should_stop


# select the second largest probability token if the argmax is the avoided token
# otherwise select the argmax token
def anything_but(probs: torch.Tensor, avoid_id: int) -> torch.Tensor:
    # probs shape (B, C)
    # return tensor shape (B, 1)
    values, indices = torch.topk(probs, 2, dim=-1)
    selected = []
    # loop over the batch dimension
    for b in range(probs.shape[0]):
        if indices[b, 0] == avoid_id:
            selected.append(indices[b, 1])
            continue
        selected.append(indices[b, 0])
    return torch.tensor(selected, dtype=torch.int32, device=probs.device).unsqueeze(1)


def validate_semantic_token_output(output: torch.Tensor) -> None:
    assert torch.all(
        (0 <= output) & (output <= SEMANTIC_VOCAB_SIZE)
    ), "unexpected output tokens"


# preprocess the texts for the generate_text_semantic model
def _preprocess_texts(texts: List[str]) -> List[str]:
    return [re.sub(r"\s+", " ", text).strip() for text in texts]


def trim_or_pad_array(
    array: Union[np.ndarray, torch.Tensor], pad_token: int, max_length: int = 256
) -> torch.Tensor:
    """
    Trim on the left (keep the right most tokens), pad on the right
    """
    # Convert np.ndarray to torch.Tensor if necessary
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).to(device=torch.device(env.DEVICE))
    else:  # Already a torch.Tensor
        tensor = array

    # Get the current length
    current_length = tensor.shape[0]

    if current_length > max_length:
        # Trim from the end (last max_length elements)
        return tensor[-max_length:]

    elif current_length < max_length:
        # Left pad 0, right pad to max_length
        padding = (0, max_length - current_length)
        return torch.nn.functional.pad(
            tensor, padding, mode="constant", value=pad_token
        )

    # If length equals max_length, just return as is
    return tensor
