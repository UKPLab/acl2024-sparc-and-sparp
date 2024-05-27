from copy import deepcopy
from functools import partial


QUERY_TEMPLATE = "{context}{question}{choices_trigger}{choices}"
RESPONSE_TEMPLATE = "{reasoning_trigger}{reasoning}{response_trigger}{response}"
PROMPT_TEMPLATE = "{system}{query}{response}"


def comma_separated_formatter(candidates):
    formatted = ""

    if isinstance(candidates, str):
        formatted = candidates

    elif candidates:  # a list
        formatted = ", ".join(candidates[:-1])

        if formatted:
            formatted += ", and "
        formatted += candidates[-1]

    return formatted


def form_query_text(
    context="",
    question="",
    choices=[],  # list of choices or already combined as string
    # set choices_trigger as empty if already part of choices as string
    choices_trigger=" The answer choices are: ",
    choices_formatter=comma_separated_formatter,
    query_former=QUERY_TEMPLATE,
):
    if choices_formatter:
        choices = choices_formatter(choices)
    if not choices:
        choices_trigger = ""

    if isinstance(query_former, str):
        query_former = query_former.format

    query = query_former(
        context=context,
        question=question,
        choices_trigger=choices_trigger,
        choices=choices,
    ).strip()
    return query


def form_response_text(
    response="",  # list of multiple answers or already combined as string
    response_trigger="",
    response_formatter=comma_separated_formatter,
    reasoning="",
    reasoning_trigger="",
    response_former=RESPONSE_TEMPLATE,
):
    if response_formatter:
        response = response_formatter(response)

    if isinstance(response_former, str):
        response_former = response_former.format

    response = response_former(
        response=response,
        response_trigger=response_trigger,
        reasoning=reasoning,
        reasoning_trigger=reasoning_trigger,
    ).strip()
    return response


def form_query_response_text(
    query="",
    response="",
    query_response_former=partial(PROMPT_TEMPLATE.format, system=""),
):
    if isinstance(query_response_former, str):
        query_response_former = query_response_former.format

    query_response = query_response_former(query=query, response=response).strip()
    return query_response


def create_prompt_messages(
    contents=[{"query": "", "response": ""}],
    user_key="query",
    model_key="response",
    role_phrase="role",
    content_phrase="content",
    user_role="user",
    model_role="assistant",
    system_content="",
    system_role="system",
    with_last_model_content=True,
):

    msgs = []
    if system_content:
        system_msg = {role_phrase: system_role, content_phrase: system_content}
        msgs.append(system_msg)

    for i, inp in enumerate(contents):
        user_msg = {role_phrase: user_role, content_phrase: inp[user_key]}
        msgs.append(user_msg)

        model_msg = inp.get(model_key)
        if model_msg and (i != len(contents) - 1 or with_last_model_content):
            model_msg = {role_phrase: model_role, content_phrase: model_msg}
            msgs.append(model_msg)

    return msgs


def create_prompt_text(
    contents=[{"query": "", "response": ""}],
    query_key="query",
    response_key="response",
    system_text="",
    query_response_combiner=form_query_response_text,
    content_sep="\n\n",
    with_last_response=True,
):

    prompt = system_text
    for i, inp in enumerate(contents):

        if i == len(contents) - 1 and not with_last_response:
            inp = deepcopy(inp)  # to avoid changing inp in source
            inp[response_key] = ""

        qr = query_response_combiner(query=inp[query_key], response=inp[response_key])

        if i != 0:
            prompt += content_sep
        prompt += qr

    prompt = prompt.strip()
    return prompt
