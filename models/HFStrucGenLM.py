import os
from functools import partial

import torch

import outlines # for structured decoding
import lmformatenforcer # lmfe library for structured decoding
from lmformatenforcer.integrations.transformers import (
    generate_enforced, 
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)

from .jsonformer import Jsonformer

from .HFGenerativeLM import (
    HF_TOKENIZE_CONFIG, 
    HF_GENERATION_CONFIG,
    HFGenerativeLM, 
)


class HFStrucGenLM(HFGenerativeLM):
    def __init__(
        self,
        parser=None, # lm-format-enforcer parser e.g. JsonSchemaParser class
        parser_schema=None,
        parser_cache_dir=".cache",
        random_state=42,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.pre_tokenize = True

        self.random_state = random_state
        self.rng = torch.Generator(device=self.model.device)
        self.rng.manual_seed(random_state)

        self._set_generate_fn(parser, parser_schema, parser_cache_dir)


    def _set_generate_fn(
        self, 
        parser=None, 
        parser_schema=None, 
        parser_cache_dir=None,
    ):
        # output parser for enforced generation using lm-format-enforcer
        self.parser_schema = parser_schema
        if isinstance(parser, str) and "json" in parser.lower():
            if hasattr(lmformatenforcer, parser):
                self.parser = getattr(lmformatenforcer, parser)(parser_schema)
                
                self.prefix_fn = build_transformers_prefix_allowed_tokens_fn(
                    self.tokenizer, self.parser
                )

                self.generate_fn = partial(
                    self.generate_fn, 
                    prefix_allowed_tokens_fn=self.prefix_fn,
                )

            elif parser == "Jsonformer":
                self.parser = parser
                self.pre_tokenize = False
                
                def jsonformer_gen(batch, **generation_config):
                    output = [
                        Jsonformer(
                            self.model, 
                            self.tokenizer, 
                            self.parser_schema, 
                            b,
                        )()
                        for b in batch
                    ]
                    return output

                self.generate_fn = jsonformer_gen

            else:
                self.parser = parser
                self.pre_tokenize = False
                
                if parser_cache_dir:
                    os.environ["OUTLINES_CACHE_DIR"] = parser_cache_dir
                
                outlines_model = outlines.models.Transformers(
                    model=self.model, tokenizer=self.tokenizer
                )
                
                generator = outlines.generate.json(outlines_model, parser_schema)
                self.generate_fn = partial(generator, rng=self.rng)


    def _generate(
        self,
        batch,
        tokenize_config=HF_TOKENIZE_CONFIG,
        generation_config=HF_GENERATION_CONFIG,
        process_text=lambda x: x.strip(),
    ):

        if self.pre_tokenize:
            output = super()._generate(
                batch, tokenize_config, generation_config, process_text
            )

        elif self.parser == "Jsonformer":
            output = self.generate_fn(batch)
            output = [[o] for o in output]

        else:
            # from outlines api for structured generation
            output = self.generate_fn(
                batch, 
                max_tokens=generation_config.get("max_new_tokens"),
                stop_at=generation_config.get("stop_at"),
            )
            output = [[o] for o in output]

        return output
