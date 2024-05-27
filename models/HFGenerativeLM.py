from copy import deepcopy

import numpy as np
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from .utils import (
    process_device_map,
    prepare_pad_token,
    prepare_chat_template,
    get_generation_count,
    generate_and_combine,
    postprocess_output_texts,
)


BNB_CONFIG = {
    "8bit": {
        "load_in_8bit": True,
        "llm_int8_threshold": 0.0,  # must be a float, otherwise throws error
    },
    "4bit": {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": getattr(torch, "bfloat16"),
    },
}

HF_TOKENIZE_CONFIG = dict(
    padding=True,
    add_special_tokens=True,
    return_tensors="pt",
    # truncation=True,
    # max_length=None,
)


"""
Recommended to either alter top_p or temperature but not both.
top_p: default=1, lesser the value, greedier the strategy
temperature: default=1, lesser the value, greedier the strategy
"""
HF_GENERATION_CONFIG = dict(
    # increase if model input is also part of model output e.g. gpt family
    max_new_tokens=64,
    top_p=1,
    temperature=1,
    do_sample=True,
    num_return_sequences=1,
)


class HFGenerativeLM:
    def __init__(
        self,
        model="meta-llama/Llama-2-7b-chat-hf",
        peft_model=[],
        tokenizer_kwargs=dict(padding_side="left"),
        pad_token=None,
        chat_template=None,
        device_map="auto",
        torch_dtype="auto",
        max_memory=None, # or dict of device-wise memory
        quantize_kwargs=dict(),  # or default key from BNB_CONFIG
        token=None,  # HF token to use for gated models
        load_model=True,  # set False to only load tokenizer, for quick testing
    ):

        if isinstance(quantize_kwargs, str):
            quantize_kwargs = BNB_CONFIG.get(quantize_kwargs, {})

        quantization_config = None
        if quantize_kwargs:

            if isinstance(quantize_kwargs.get("bnb_4bit_compute_dtype"), str):
                quantize_kwargs["bnb_4bit_compute_dtype"] = getattr(
                    torch, quantize_kwargs["bnb_4bit_compute_dtype"]
                )

            quantization_config = BitsAndBytesConfig(**quantize_kwargs)

        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)
        if not torch.cuda.is_available():
            # less than full precision dtypes not supported on cpu yet
            torch_dtype = torch.float32

        TokenizerClass = AutoTokenizer
        ModelClass = AutoModelForCausalLM
        self.output_has_input = True

        if "t5" in model:
            self.output_has_input = False
            TokenizerClass = T5Tokenizer
            ModelClass = T5ForConditionalGeneration

        elif "gpt2" in model:
            TokenizerClass = GPT2Tokenizer
            ModelClass = GPT2LMHeadModel

        self.tokenizer = TokenizerClass.from_pretrained(
            model,
            **tokenizer_kwargs,
        )

        device_map = process_device_map(
            device_map,
            model=model,
            max_memory=max_memory,
            dtype=None if torch_dtype == "auto" else torch_dtype,
        )

        self.base_model = None
        if load_model:
            self.base_model = ModelClass.from_pretrained(
                model,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                token=token,
            )

        # add pad token if not present for batched inference
        if not self.tokenizer.pad_token:
            self.tokenizer, self.base_model = prepare_pad_token(
                self.tokenizer, self.base_model, pad_token=pad_token
            )

        self.tokenizer = prepare_chat_template(self.tokenizer, chat_template)

        self.model = self.base_model
        if peft_model and load_model:
            if isinstance(peft_model, str):
                peft_model = [peft_model]

            for ft_model in peft_model:
                self.model = PeftModel.from_pretrained(self.model, ft_model)
                self.model = self.model.merge_and_unload()

        # can be set differently in any child class
        self.generate_fn = getattr(self.model, "generate", None)


    def prepare_batch(self, batch):
        is_chat = False
        if batch:
            if isinstance(batch, str):
                batch = [batch]

            elif isinstance(batch, list) and isinstance(batch[0], dict):
                is_chat = True
                batch = [self.tokenizer.apply_chat_template(batch, tokenize=False)]

            elif (
                isinstance(batch, list)
                and isinstance(batch[0], list)
                and isinstance(batch[0][0], dict)
            ):
                is_chat = True
                batch = [
                    self.tokenizer.apply_chat_template(chat, tokenize=False)
                    for chat in batch
                ]

        return batch, is_chat


    def _generate(
        self,
        batch,
        tokenize_config=HF_TOKENIZE_CONFIG,
        generation_config=HF_GENERATION_CONFIG,
        process_text=lambda x: x.strip(),
    ):

        tokenized = self.tokenizer(batch, **tokenize_config)
        tokenized = tokenized.to(self.model.device)

        self.model.eval()
        with torch.no_grad():
                # output = self.model.generate(**tokenized, **generation_config)
                output = self.generate_fn(**tokenized, **generation_config)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        # extract decoded batch without special tokens
        # to perform correct start text removal when output has input
        batch = self.tokenizer.batch_decode(
            tokenized.input_ids, skip_special_tokens=True
        )

        output = postprocess_output_texts(
            output_texts=output,
            remove_start_texts=batch if self.output_has_input else None,
            n_sequence_per_input=generation_config.get("num_return_sequences", 1),
            process_text=process_text,
        )

        return output


    def generate(
        self,
        batch,
        total_generation=1,
        tokenize_config=HF_TOKENIZE_CONFIG,
        generation_config=HF_GENERATION_CONFIG,
        process_text=lambda x: x.strip(),
    ):
        r"""
        effective number of return sequences:
            num_regeneration x num_return_sequences (from generation_config)
        increase num_regeneration and decrease num_return_sequences
        to work with less memory
        """

        batch, is_chat = self.prepare_batch(batch)

        if is_chat and tokenize_config.get("add_special_tokens", True):
            # special tokens already added by chat template
            tokenize_config = deepcopy(tokenize_config)
            tokenize_config["add_special_tokens"] = False

        num_return_sequences = generation_config.get("num_return_sequences", 1)

        generation_count = get_generation_count(
            total_generation=total_generation,
            num_return_sequences=num_return_sequences,
        )

        output = generate_and_combine(
            self._generate,
            batch,
            generation_count=generation_count,
            total_generation=total_generation,
            generation_config=generation_config,
            process_text=process_text,
        )

        return output
