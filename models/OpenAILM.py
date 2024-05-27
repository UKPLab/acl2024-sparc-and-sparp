import re
import os
import json
import time
from functools import partial
from copy import deepcopy

import openai

from .utils import (
    get_generation_count,
    generate_and_combine,
    postprocess_output_texts,
)


OPENAI_CLIENT_CONFIG = {
    "api_version": "2023-05-15",
    "api_key": os.getenv("AZURE_OPENAI_KEY"),
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
}

"""
https://platform.openai.com/docs/api-reference/chat
Recommended to either alter top_p or temperature but not both.
top_p: default=1, min=0, max=1, lesser the value, greedier the strategy
temperature: default=1, min=0, max=2, lesser the value, greedier the strategy
"""
OPENAI_GENERATION_CONFIG = dict(
    model="text-davinci-003",
    n=1,  # number of completions
    max_tokens=64,  # tokens in model response, high for cot, low otherwise e.g. 32
    top_p=1,  # sample from tokens percentile
    temperature=1,  # modifies proba values
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,  # or ["\n"] in case of code-*-* models e.g. code-davinci-002
)


class OpenAILM:
    def __init__(self, client="AzureOpenAI", config=OPENAI_CLIENT_CONFIG):

        self.model = getattr(openai, client)(**config)
        self.reset()

        self.blank_chat_output = {
            "choices": [
                {
                    "finish_reason": "",
                    "index": 0,
                    "message": {
                        "content": "",
                        "role": "assistant",
                        "function_call": None,
                        "tool_calls": None,
                    },
                    "logprobs": None,
                },
            ],
            "created": 0,
            "id": "",
            "model": "",
            "object": "",
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
            "system_fingerprint": None,
        }

    def reset(self):
        self.stored_output = []

    def _error_as_output_handler(
        self,
        error,
        model="gpt-4",
        n=1,
    ):
        error_str = repr(error)

        print(f"{error_str}\nContinuing by omitting and outputting error as content.")

        output = deepcopy(self.blank_chat_output)
        output["model"] = model

        choice = output["choices"][0]
        choice["finish_reason"] = "error"
        choice["message"]["content"] = error_str

        output["choices"] = [choice for _ in range(n)]

        return output

    def _complete_chat(
        self,
        messages,
        generation_config=OPENAI_GENERATION_CONFIG,
        dump_json=True,
    ):
        output = self.model.chat.completions.create(
            messages=messages, **generation_config
        )

        if dump_json:
            output = json.loads(output.model_dump_json())
        return output

    def _complete_text(
        self,
        batch,
        generation_config=OPENAI_GENERATION_CONFIG,
        dump_json=True,
    ):
        output = self.model.completions.create(prompt=batch, **generation_config)
        if dump_json:
            output = json.loads(output.model_dump_json())
        return output

    def _generate(
        self,
        batch,
        is_chat_completion=True,
        generation_config=OPENAI_GENERATION_CONFIG,
        process_text=lambda x: x.strip(),
        store_output=True,
    ):

        if is_chat_completion:
            output = []

            for messages in batch:

                try:
                    out = self._complete_chat(
                        error_handlers=error_handlers,
                        messages=messages,
                        generation_config=generation_config,
                    )

                except openai.BadRequestError as err:
                    out = self._error_as_output_handler(
                            err, 
                            model=generation_config.get("model", "gpt-35-turbo"),
                            n=generation_config.get("n", 1),
                        )

                output.append(out)

            output_texts = [
                o["message"].get("content", "")
                for out in output
                for o in out["choices"]
            ]

        else:
            output = self._complete_text(
                batch=batch,
                generation_config=generation_config,
            )

            output_texts = [o["text"] for o in output["choices"]]

        if store_output:
            self.stored_output.extend(output)

        output = postprocess_output_texts(
            output_texts=output_texts,
            remove_start_texts=None,  # pass if needs to be removed from output
            n_sequence_per_input=generation_config.get("n", 1),
            process_text=process_text,
        )

        return output

    def generate(
        self,
        batch,
        total_generation=1,
        generation_config=OPENAI_GENERATION_CONFIG,
        process_text=lambda x: x.strip(),
        store_output=True,
    ):

        if isinstance(batch, str) or (
            isinstance(batch, list) and isinstance(batch[0], dict)
        ):
            batch = [batch]

        model = generation_config.get("model", "gpt-35-turbo")
        n = generation_config.get("n", 1)
        is_chat_completion = "gpt-35-turbo" in model or "gpt-4" in model

        generation_count = get_generation_count(
            total_generation=total_generation, num_return_sequences=n
        )

        output = generate_and_combine(
            self._generate,
            batch,
            generation_count=generation_count,
            total_generation=total_generation,
            is_chat_completion=is_chat_completion,
            generation_config=generation_config,
            process_text=process_text,
            store_output=store_output,
        )

        return output
