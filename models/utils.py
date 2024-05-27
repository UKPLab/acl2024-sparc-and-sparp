from functools import partial

import numpy as np
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
)

from trl import is_xpu_available

from .templates import llama_chat_template


def prepare_pad_token(tokenizer, model=None, pad_token="unk_token"):
    pad_token = tokenizer.special_tokens_map.get(pad_token, pad_token)
    if pad_token:
        tokenizer.add_special_tokens({"pad_token": pad_token})

        if model:
            if model.vocab_size != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))

            model.config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        print(f"Pad token added:")
        print(
            f"pad_token: {tokenizer.pad_token}\n"
            f"pad_token_id: {tokenizer.pad_token_id}"
        )

    return tokenizer, model


def prepare_chat_template(tokenizer, chat_template=None):
    if chat_template:
        tokenizer.chat_template = chat_template
    elif not tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.default_chat_template
        if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            # rectify incorrect spacing between eos and bos
            tokenizer.chat_template = partial(llama_chat_template, tokenizer)()
    return tokenizer


def process_device_map(device_map="auto", model=None, **kwargs):
    r"""
    Currently supports only Causal LM when model is specified for device_map=infer
    if distributed data parallel, then copy the model to each device
    taken from trl's sft example script:
    https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
    """

    if device_map == "ddp":

        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )

    elif device_map == "infer" and model:
        config = AutoConfig.from_pretrained(model)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        device_map = infer_auto_device_map(
            model, no_split_module_classes=model._no_split_modules, **kwargs
        )

    return device_map


def get_generation_count(total_generation=1, num_return_sequences=1):

    if total_generation < num_return_sequences:
        print(
            f"`total_generation` < `num_return_sequences`."
            "Setting it equal to `num_return_sequences`"
        )
        total_generation = num_return_sequences

    generation_count = int(np.ceil(total_generation / num_return_sequences))

    return generation_count


def generate_and_combine(
    gen_fn,
    batch,
    generation_count,
    total_generation,
    **gen_kwargs,
):
    output = [[] for i in range(len(batch))]

    for i in range(generation_count):
        cur_out = gen_fn(batch=batch, **gen_kwargs)

        # append subsequent regenerations
        output = [all_out + out for all_out, out in zip(output, cur_out)]

    # discard extra generation in the last generation_count
    output = [i[:total_generation] for i in output]

    return output


def postprocess_output_texts(
    output_texts,
    remove_start_texts=None,  # pass if needs to be removed from output
    n_sequence_per_input=1,
    process_text=lambda x: x.strip(),
):
    num_inputs = len(output_texts) // n_sequence_per_input
    processed = [[] for _ in range(num_inputs)]

    for i, t in enumerate(output_texts):
        if remove_start_texts:
            text_to_remove = remove_start_texts[i // n_sequence_per_input]
            t = t[len(text_to_remove) :]
        t = process_text(t)
        processed[i // n_sequence_per_input].append(t)

    return processed
