from transformers.models.llama import tokenization_llama_fast


DEFAULT_SYSTEM_PROMPT = tokenization_llama_fast.DEFAULT_SYSTEM_PROMPT

# @property
# Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.default_chat_template
def llama_chat_template(self):
    """
    LLaMA uses [INST] and [/INST] to indicate user messages, and <<SYS>> and <</SYS>> to indicate system messages.
    Assistant messages do not have special tokens, because LLaMA chat models are generally trained with strict
    user/assistant/user/assistant message ordering, and so assistant messages can be identified from the ordering
    rather than needing special tokens. The system message is partly 'embedded' in the first user message, which
    results in an unusual token ordering when it is present. This template should definitely be changed if you wish
    to fine-tune a model with more flexible role ordering!

    The output should look something like:

    <bos> [INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer <eos> <bos> [INST] Prompt [/INST] Answer <eos>
    <bos> [INST] Prompt [/INST]

    The reference for this chat template is [this code
    snippet](https://github.com/facebookresearch/llama/blob/556949fdfb72da27c2f4a40b7f0e4cf0b8153a28/llama/generation.py#L320-L362)
    in the original repository.
    """

    # logger.warning_once(
    #     "\nNo chat template is defined for this tokenizer - using the default template "
    #     f"for the {self.__class__.__name__} class. If the default is not appropriate for "
    #     "your model, please set `tokenizer.chat_template` to an appropriate template. "
    #     "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
    # )

    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
        "{% set system_message = messages[0]['content'] %}"
        "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
        "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
        "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = false %}"
        "{% endif %}"
        "{% for message in loop_messages %}"  # Loop over all non-system messages
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
        "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
        "{% else %}"
        "{% set content = message['content'] %}"
        "{% endif %}"
        "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
        "{{ bos_token + ' [INST] ' + content.strip() + ' [/INST]' }}"  # corrective space before [INST]
        "{% elif message['role'] == 'system' %}"
        "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ ' '  + content.strip() + ' ' + eos_token }}"
        "{% if loop.revindex0 != 0 %}"  # correction to add space between bos and eos
        "{{ ' ' }}"  # correction to add space between bos and eos
        "{% endif %}"  # correction to add space between bos and eos
        "{% endif %}"
        "{% endfor %}"
    )
    template = template.replace(
        "USE_DEFAULT_PROMPT", "true" if self.use_default_system_prompt else "false"
    )
    default_message = DEFAULT_SYSTEM_PROMPT.replace("\n", "\\n").replace("'", "\\'")
    template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

    return template
