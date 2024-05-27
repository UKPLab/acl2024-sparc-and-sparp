import transformers
from transformers import EarlyStoppingCallback
from transformers.integrations import NeptuneCallback
import trl


def prepare_callbacks(callbacks, run=None):
    for i in range(len(callbacks)):
        if callbacks[i]["name"] == "NeptuneCallback" and run:
            callbacks[i] = NeptuneCallback(run=run)
        elif callbacks[i]["name"] == "EarlyStoppingCallback":
            callbacks[i] = EarlyStoppingCallback(**callbacks[i]["args"])

    return callbacks


def train_clm(
    model,
    tokenizer,
    train_dataset,
    trainer_class="SFTTrainer",
    args_class="TrainingArguments",
    eval_dataset=None,
    save_model=None,
    training_args=dict(
        output_dir="",
        num_train_epochs=3,
        # max_steps=-1, # Number of training steps (overrides num_train_epochs)
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=None,  # leave None as default
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=100,
        save_total_limit=10,
        save_strategy="steps",
        # # evaluation parameters, uncomment to use
        # per_device_eval_batch_size=4,
        # eval_steps=250,
        # evaluation_strategy="steps",
        learning_rate=1e-4,
        weight_decay=0.001,  # Weight decay to apply to all layers except bias/LayerNorm weights
        # fp16=True, # either one of this or bf16 or none of them at a time
        bf16=True,
        max_grad_norm=0.3,  # Maximum gradient normal (gradient clipping)
        warmup_ratio=0.03,  # Ratio of steps for a linear warmup (from 0 to learning rate)
        # Group sequences into batches with same length
        # Saves memory and speeds up training considerably
        group_by_length=True,
        lr_scheduler_type="cosine",
        # report_to="tensorboard",
        report_to="none",
        # ddp_find_unused_parameters=False,
    ),
    callbacks=[],
    trainer_args=dict(
        dataset_text_field="text",  # or else use instead formatting_func
        # peft_config=None,
        # max_seq_length: # default none is same as 1024
        max_seq_length=2048,
        packing=False,
        neftune_noise_alpha=5,
    ),
    run=None,
):

    if training_args.get("gradient_checkpointing", True):
        # gradient checkpointing is True either explicitly or by default
        # then disable model caching as it is incompatible for training.
        model.config.use_cache = False

    # enable HF experimental tensor parallelism (tp)
    model.config.pretraining_tp = 1

    if hasattr(trl, args_class):
        training_arguments = getattr(trl, args_class)(**training_args)
    else:
        training_arguments = getattr(transformers, args_class)(**training_args)

    trainer = getattr(trl, trainer_class)(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        callbacks=prepare_callbacks(callbacks, run=run),
        **trainer_args,
    )

    trainer.train()

    if save_model:
        trainer.model.save_pretrained(save_model)

    return trainer
