import colorama
from tqdm.auto import tqdm
from accelerate import Accelerator
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import load_dataset


def get_top_ks(model, dataset, data_indices, universal_prompt, top_k):
    """Code to get top-k substitutions according to gradients"""
    # Creating one-hot encoding for the suffix (to get gradients)
    one_hot = torch.zeros(
        (1, universal_prompt.shape[1], model.config.vocab_size),
        device=model.device,
        requires_grad=True,
        dtype=model.dtype,
    )
    for i in range(universal_prompt.shape[1]):
        one_hot.data[0, i, universal_prompt[0, i]] = 1

    # Collecting top-k substitutions
    top_ks = []
    for idx in tqdm(data_indices, desc="Getting top-k substitutions...", leave=False):
        # Getting sample
        sample = dataset[idx]
        inputs = {
            k: torch.tensor(v, device=model.device) for k, v in sample["inputs"].items()
        }
        ss, es = (
            sample["indices"]["suffix_start_idx"],
            sample["indices"]["suffix_end_idx"],
        )

        # Getting input embeds
        input_embeds = model.get_input_embeddings()(inputs["input_ids"])
        input_embeds[:, ss:es] = one_hot @ model.get_input_embeddings().weight

        # Getting gradients
        inputs["inputs_embeds"] = input_embeds
        del inputs["input_ids"]
        compute_loss(model, inputs).backward()
        gradients = -one_hot.grad
        one_hot.grad = None  # Zeroing to not interfere with next sample

        # Getting top-k substitutions
        top_ks.append(torch.topk(gradients[0], k=top_k, dim=-1).indices)

    return torch.stack(top_ks)


@torch.inference_mode()
def get_losses(model, dataset, data_indices, universal_prompt, top_ks):
    """Code to get the losses for all samples given top-k substitutions"""
    losses = []

    sub_indices = np.random.randint(0, universal_prompt.shape[1], len(data_indices))
    sub_ks = np.random.randint(0, top_ks.shape[-1], len(data_indices))

    item = 0
    for idx, sub_idx, sub_k in tqdm(
        zip(data_indices, sub_indices, sub_ks), desc="Getting losses...", leave=False
    ):
        # Getting sample
        sample = dataset[idx]
        ss, es = (
            sample["indices"]["suffix_start_idx"],
            sample["indices"]["suffix_end_idx"],
        )

        # Modifying initial suffix with universal prompt + substitution
        inputs = {
            k: torch.tensor(v, device=model.device) for k, v in sample["inputs"].items()
        }
        inputs["input_ids"][:, ss:es] = universal_prompt
        sub_token = top_ks[item, sub_idx, sub_k]
        inputs["input_ids"][:, sub_idx] = sub_token
        item += 1

        # Computing loss
        loss = compute_loss(model, inputs)
        losses.append((loss.cpu(), sub_idx, sub_token))
    return losses


def compute_loss(model, inputs):
    # Method to compute the loss given model and its inputs
    return model(**inputs).loss


def main():
    set_seed(0)

    # Model parameters
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Attack parameters
    batch_size = 512  # Number of samples to optimize over (512 in GCG paper)
    top_k = 256  # Number of top tokens to sample from (256 in GCG paper)
    steps = 500  # Total number of optimization steps (500 in GCG paper)
    suffix_length = 20  # Length of the suffix to be optimized (20 in GCG paper)
    suffix_initial_token = " !"  # Initial token repeated for the length of the suffix
    system_prompt = ""  # System prompt to be prepended to the input

    # Initial suffix
    initial_suffix = suffix_initial_token * suffix_length

    # Loading model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Getting suffix ids
    initial_suffix_ids = tokenizer.encode(
        initial_suffix, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    assert initial_suffix_ids.shape[1] == suffix_length, (
        f"Initial suffix length {initial_suffix_ids.shape[1]} does not match expected length {suffix_length}."
    )

    # Loading dataset
    dataset = load_dataset("walledai/AdvBench", split="train")

    # Tokenizing dataset
    def tokenize(sample):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"] + initial_suffix},
            {"role": "assistant", "content": sample["target"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        inputs = tokenizer(text, return_tensors="pt")
        ids_list = inputs["input_ids"].clone()[0].tolist()

        # Finding start and end indices for suffix and target response
        suffix_start_idx = ids_list.index(initial_suffix_ids[0, 0])
        suffix_end_idx = suffix_start_idx + suffix_length

        initial_response_id = tokenizer.encode(
            sample["target"], return_tensors="pt", add_special_tokens=False
        )[0, 0]
        target_start_idx = suffix_end_idx + ids_list[suffix_end_idx:].index(
            initial_response_id
        )
        target_end_idx = len(ids_list) - 1

        # Creating labels
        labels = torch.ones_like(inputs["input_ids"]) * -100
        labels[:, target_start_idx:target_end_idx] = inputs["input_ids"][
            :, target_start_idx:target_end_idx
        ]
        inputs["labels"] = labels

        return {
            "inputs": inputs,
            "indices": {
                "suffix_start_idx": suffix_start_idx,
                "suffix_end_idx": suffix_end_idx,
                "target_start_idx": target_start_idx,
                "target_end_idx": target_end_idx,
            },
        }

    dataset = dataset.map(tokenize, load_from_cache_file=False, batched=False)

    # Moving model to device
    acc = Accelerator()
    model = acc.prepare(model)

    # Showing legend
    print(
        colorama.Fore.YELLOW
        + "INITIAL"
        + colorama.Style.RESET_ALL
        + " - Untoched tokens w.r.t initial suffix"
    )
    print(
        colorama.Fore.GREEN
        + "MODIFIED"
        + colorama.Style.RESET_ALL
        + " - Modified tokens w.r.t initial suffix"
    )
    print(
        colorama.Fore.RED
        + "CURRENT"
        + colorama.Style.RESET_ALL
        + " - Current token we try to modify\n\n"
    )

    # Optimizing universal prompt
    # NOTE: Each step takes ~47s on an RTX 4090 GPU, 4-bit quantized LLama-3.2-3B model, batch size 512, top-k 256, bfloat16 compute dtype
    universal_prompt = initial_suffix_ids.clone()
    data_indices = list(range(min(batch_size, len(dataset))))
    for step in tqdm(range(steps), desc="Optimizing universal prompt"):
        # Obtaining top-k for all samples
        top_ks = get_top_ks(
            model, dataset, data_indices, universal_prompt, top_k
        )  # (B, Suffix length, K)

        # Evaluating losses for substitutions
        losses = get_losses(
            model, dataset, data_indices, universal_prompt, top_ks
        )  # {(loss, position, token_id)}
        mean_loss = np.mean([el[0] for el in losses])

        # Picking substitution with minimum loss
        min_loss_idx = np.argmin([el[0] for el in losses])

        # Updating global perturbation
        best_position, best_token_id = losses[min_loss_idx][1], losses[min_loss_idx][2]
        universal_prompt[:, best_position] = best_token_id

        # Logging
        suffix_str = ""
        suffix_text = ""
        for i, tok_id in enumerate(universal_prompt[0].tolist()):
            if i == best_position:
                suffix_str += colorama.Fore.RED + str(tok_id)
                suffix_text += colorama.Fore.RED + tokenizer.decode(
                    tok_id, add_special_tokens=False
                )
            elif tok_id == initial_suffix_ids[0, 0]:
                suffix_str += colorama.Fore.YELLOW + str(tok_id)
                suffix_text += colorama.Fore.YELLOW + tokenizer.decode(
                    tok_id, add_special_tokens=False
                )
            else:
                suffix_str += colorama.Fore.GREEN + str(tok_id)
                suffix_text += colorama.Fore.GREEN + tokenizer.decode(
                    tok_id, add_special_tokens=False
                )
            suffix_str += colorama.Style.RESET_ALL + " "
            suffix_text += colorama.Style.RESET_ALL + " "

        print(f"Mean loss: {mean_loss:.2f}")
        print(f"Current universal prompt  (ids): {suffix_str}")
        print(f"Current universal prompt (text): {suffix_text}")
        print("\n\n")


if __name__ == "__main__":
    main()
