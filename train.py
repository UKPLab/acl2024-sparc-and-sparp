import yaml
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

import peft
from datasets import concatenate_datasets

from data.utils import (
    load_data,
    shuffle_select_data,
    create_chat_templated_text_data,
)
import models as GenerativeLM
from models.train import train_clm
from utils.log import init_neptune, get_gpu_info, save_and_upload_output


SEED = 42


def main(exp, run=None):

    dataset = load_data(**exp["data"]["load"])

    dataset = {
        name: shuffle_select_data(
            data,
            seed=exp.get("seed", SEED),
            shuffle=exp["data"].get("shuffle"),
            select=exp["data"].get("select"),  # None for whole dataset
        )[0]  # get the dataset but leave out the selection info
        for name, data in dataset.items()
    }

    print(f"\nGPU info before model load:\n{get_gpu_info()}\n")

    model = getattr(GenerativeLM, exp["model"]["model_class"])(
        **exp["model"]["init_args"]
    )

    print(f"\nGPU info after model load:\n{get_gpu_info()}\n")

    dataset = create_chat_templated_text_data(
        dataset,
        chat_template_func=model.tokenizer.apply_chat_template,
        **exp["data"]["chat_template_args"],
    )

    dataset = {
        split: concatenate_datasets([data[split] for data in dataset.values()])
        for split in exp["data"]["chat_template_args"].get("prepare_splits", ["train"])
    }

    training_args = exp["train"]["training"]
    training_args["output_dir"] = exp["output_dir"]

    trainer_args = exp["train"].get("trainer", {})
    if peft_config := exp.get("peft"):
        peft_config = getattr(peft, peft_config["config"])(**peft_config["config_args"])
    trainer_args["peft_config"] = peft_config

    trainer = train_clm(
        model=model.model,
        tokenizer=model.tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        save_model=exp["output_dir"] / "final",
        callbacks=exp["train"].get("callbacks", []),
        training_args=training_args,
        trainer_args=trainer_args,
        run=run,
    )


def process_config_for_log(config):
    save_dir = Path(config["exp"]["output_dir"])

    model_name = config["exp"]["model"]["model_class"]
    if model_name == "OpenAILM":
        model_name = "openai"
    else:
        model_name = config["exp"]["model"]["init_args"]["model"]

    model_name = model_name.replace("/", "-")
    if model_name == "openai":
        model_name += "-" + config["exp"]["model"]["generation_config"]["model"]

    save_dir = save_dir / f"train-{model_name}-{str(datetime.now())}"
    config["exp"]["output_dir"] = save_dir
    return config


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Supervised finetuning of LLMs on spatial reasoning datasets."
    )

    parser.add_argument(
        "--config",
        default="config/train-config.yaml",
        help="path to config file specifying parameters of training.",
    )

    parser.add_argument(
        "--private",
        default="config/private-config.yaml",
        help="path to private file specifying api keys and tokens.",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.private) as f:
        private = yaml.safe_load(f)

    run = init_neptune(config.get("neptune"), private.get("neptune-api-token"))
    if run:
        run["reference/config.yaml"].upload(args.config)

    config = process_config_for_log(config)
    main(exp=config["exp"], run=run)
