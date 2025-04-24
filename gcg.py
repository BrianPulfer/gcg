import colorama
from tqdm import tqdm
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

set_seed(0)

# Attack parameters
steps = 500  # Number of optimization steps for a single sample
max_samples = 5  # Number of samples to optimize
suffix_length = 20  # Length of the suffix to be optimized
suffix_initial_token = " !"  # Initial token repeated for the length of the suffix
system_prompt = ""  # System prompt to be prepended to the input
target_response = "Sure, here's"  # Target response we want the model to generate
top_k = 256  # Number of top tokens to sample from (usually 256)


def gcg(model, tokenizer, sample, acc, steps, verbose=True, n_logs=10):
    inputs = {k: torch.tensor(v).to(acc.device) for k, v in sample["inputs"].items()}
    input_ids = inputs["input_ids"]
    del inputs["input_ids"]

    # Creating one-hot encoding for the suffix (to get gradients)
    one_hot = torch.zeros(
        (1, suffix_length, model.config.vocab_size),
        device=acc.device,
        requires_grad=True,
        dtype=model.dtype,
    )
    suffix_initial_token_id = tokenizer.encode(
        suffix_initial_token, add_special_tokens=False
    )[0]
    one_hot.data[:, :, suffix_initial_token_id] = 1

    # Getting indices for prefix (what to optimize) and answer (logits on which to compute loss)
    indices = sample["indices"]
    ss, es = indices["suffix_start_idx"], indices["suffix_end_idx"]
    st, et = indices["target_start_idx"], indices["target_end_idx"]

    for step in tqdm(range(steps), desc="Attacking sample...", leave=False):
        # Getting input embeds
        input_embeds = model.get_input_embeddings()(input_ids)
        suffix_embeds = (
            one_hot @ model.get_input_embeddings().weight
        )  # To get gradients w.r.t. one-hot encoding
        input_embeds[:, ss:es] = suffix_embeds

        # Changing input_ids to input_embeds
        inputs["inputs_embeds"] = input_embeds

        # Getting loss and gradients
        # loss = model(**inputs).loss
        logits = model(**inputs).logits
        loss = torch.nn.functional.cross_entropy(
            logits[0, st:et, :],
            inputs["labels"][0, st:et],
            reduction="mean",
        )
        loss.backward()

        # Getting gradients
        gradients = -one_hot.grad

        # Trying substitution for a random token in the suffix (among top-k others)
        sub_idx = np.random.randint(0, suffix_length)
        topk = torch.topk(gradients[0, sub_idx], k=top_k, dim=-1).indices

        # Checking if loss decreases for any of those
        one_hot_copies = torch.zeros(
            top_k,
            suffix_length,
            model.config.vocab_size,
            device=acc.device,
            dtype=model.dtype,
        )
        for i, token in enumerate(topk):
            one_hot_copies[i, sub_idx, token] = 1

        # Getting embeds
        sub_input_embeds = one_hot_copies @ model.get_input_embeddings().weight
        new_input_embeds = input_embeds.clone().repeat(top_k, 1, 1)
        new_input_embeds[:, ss:es] = sub_input_embeds

        # Getting loss for all the top-k substitutions
        with torch.no_grad():
            # Changing input embeds to all top-k substitutions
            inputs["inputs_embeds"] = new_input_embeds

            # Removing labels from inputs
            labels = inputs["labels"]
            del inputs["labels"]

            # Computing best loss
            logits = model(**inputs).logits
            losses = [
                torch.nn.functional.cross_entropy(
                    logits[i, st:et, :],
                    labels[0, st:et],
                    reduction="mean",
                ).item()
                for i in range(top_k)
            ]
            best_sub_idx = np.argmin(losses)

            # If loss decreases, substitute the token
            if losses[best_sub_idx] < loss.item():
                one_hot[0, sub_idx, :] = 0  # Zeroing past one-hot encoding
                one_hot[0, sub_idx, topk[best_sub_idx]] = 1  # Substituting the token

            # Putting labels back
            inputs["labels"] = labels

        # Logging loss
        if verbose:
            if step % (steps // n_logs) == 0:
                curr_suffix_ids = one_hot[0].argmax(dim=-1).tolist()
                suffix_str = " ".join(
                    [
                        f"{colorama.Fore.YELLOW if id == suffix_initial_token_id else colorama.Fore.GREEN} {id} {colorama.Fore.RESET}"
                        for id in curr_suffix_ids
                    ]
                )
                print(f"Step {step + 1} / {steps} Loss: {loss.item():.4f} {suffix_str}")
    # Obtaining final input ids
    new_input_ids = one_hot.argmax(dim=-1)
    input_ids[:, ss:es] = new_input_ids
    return input_ids[:, :st]


def main():
    # Model parameters
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Initial suffix
    initial_suffix = suffix_initial_token * suffix_length

    # Loading model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Loading dataset
    dataset = load_dataset("walledai/AdvBench", split="train")

    # Sys prompt | Question | Suffix | Target answer
    #   suffix_start_idx - suffix_end_idx
    #   target_start_idx - target_end_idx

    # Tokenizing dataset
    def tokenize(sample):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"] + initial_suffix},
            {"role": "assistant", "content": target_response},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        inputs = tokenizer(text, return_tensors="pt")

        # Finding start and end indices for suffix and response
        suffix_start_idx = (
            inputs["input_ids"][0]
            .tolist()
            .index(tokenizer.encode(initial_suffix, add_special_tokens=False)[0])
        )
        suffix_end_idx = suffix_start_idx + len(
            tokenizer.encode(initial_suffix, add_special_tokens=False)
        )
        target_start_idx = suffix_end_idx + inputs["input_ids"][0][
            suffix_end_idx:
        ].tolist().index(tokenizer.encode(target_response, add_special_tokens=False)[0])
        target_end_idx = target_start_idx + len(
            tokenizer.encode(target_response, add_special_tokens=False)
        )

        # Creating labels
        labels = torch.ones_like(inputs["input_ids"]) * -100
        labels[:, target_start_idx:target_end_idx] = inputs["input_ids"][
            :, target_start_idx:target_end_idx
        ]
        inputs["labels"] = labels

        return {
            "prompt": sample["prompt"],
            "target": sample["target"],
            "inputs": inputs,
            "indices": {
                "suffix_start_idx": suffix_start_idx,
                "suffix_end_idx": suffix_end_idx,
                "target_start_idx": target_start_idx,
                "target_end_idx": target_end_idx,
            },
        }

    dataset = dataset.map(
        tokenize,
        remove_columns=["prompt", "target"],
        load_from_cache_file=False,
        batched=False,
    )

    # Moving model to device
    acc = Accelerator()
    model = acc.prepare(model)

    # Running optimization over dataset
    for sample_idx in tqdm(
        range(min(max_samples, len(dataset))), desc="Attacking dataset..."
    ):
        # Getting sample
        sample = dataset[sample_idx]

        # Running GCG to get new input ids
        new_input_ids = gcg(
            model, tokenizer, sample, acc, steps=steps, verbose=True, n_logs=10
        )

        # Evaluating output of model
        with torch.no_grad():
            out = model.generate(
                input_ids=new_input_ids, do_sample=False, max_new_tokens=100
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"Sample {sample_idx + 1}/{len(dataset)}:")
            print(text)


if __name__ == "__main__":
    main()
