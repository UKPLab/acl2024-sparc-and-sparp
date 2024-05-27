import re
import json
from copy import deepcopy

from lxml import etree


JSON_BLOCK_PAT = re.compile("(?<=```json).*(?=```)", flags=re.DOTALL)
# Dangling comma in json objects needs to be removed beforehand
# e.g. json.loads on '{"a": "b",}' throws error because of dangling comma
DANGLING_COMMA_PAT = re.compile(",\n*\s*(?=}|]|$)")

CODE_BLOCK_TEMPLATE = """```{block_name}
{block}
```"""

ENCLOSER_HOLDER = "extraction"
ENCLOSER_ENDER = "extraction_status"
JSON_ENCLOSER = {ENCLOSER_HOLDER: {}, ENCLOSER_ENDER: "Done"}

XML_DTYPE_TO_VALUE = {
    "string": "",
    "integer": 0,
    "float": 0.0,
    "list": [],
    "object": {},
}


def texts_to_jsons(
    texts,
    lookbehind_regex=f'"{ENCLOSER_HOLDER}":',  # often with a dummy lookbehind placed in generation
    lookahead_regex=f'"{ENCLOSER_ENDER}"',  # often with a dummy lookahead placed in generation
):
    if isinstance(texts, str):
        texts = [texts]

    pattern = ".*"
    if lookbehind_regex:
        lookbehind_regex = f"(?<={lookbehind_regex})"
        pattern = lookbehind_regex + pattern

    if lookahead_regex:
        lookahead_regex = f"(?={lookahead_regex})"
        pattern += lookahead_regex

    if pattern != ".*":
        lookup_pat = re.compile(pattern, flags=re.DOTALL)
    else:
        lookup_pat = JSON_BLOCK_PAT

    json_objs = []
    for i, txt in enumerate(texts):
        json_objs.append([])
        json_block = lookup_pat.findall(txt)
        if not json_block:
            # attempt direct conversion of txt
            json_block.append(txt)

        for block in json_block:
            block = DANGLING_COMMA_PAT.sub("", block)
            try:
                json_objs[i].append(json.loads(block))
            except json.JSONDecodeError as e:
                print(repr(e), "Continuing by appending an empty JSON object.")
                json_objs[i].append({})

    if len(json_objs) == 1:
        json_objs = json_objs[0]

    return json_objs


def enclose_in_code_blocks(texts, block_name="json"):
    if isinstance(texts, str):
        texts = [texts]

    code_blocks = [
        CODE_BLOCK_TEMPLATE.format(block_name=block_name, block=txt) for txt in texts
    ]

    if len(code_blocks) == 1:
        code_blocks = code_blocks[0]

    return code_blocks


def enclose_in_json(
    values,
    encloser_json=JSON_ENCLOSER,
    encloser_key=ENCLOSER_HOLDER,
    make_copy=False,
):

    if not isinstance(values, list):
        values = [values]

    enclosed_jsons = []
    for v in values:
        if make_copy:
            v = deepcopy(v)
        encloser = deepcopy(encloser_json)
        encloser[encloser_key] = v
        enclosed_jsons.append(encloser)

    if len(enclosed_jsons) == 1:
        enclosed_jsons = enclosed_jsons[0]

    return enclosed_jsons


def json_template_from_xml(xml_input, is_file=True, key_attrib="name"):
    if is_file:
        xml_node = etree.parse(xml_input).getroot()
    else:
        xml_node = etree.fromstring(xml_input)

    json_template = {}
    node_stack = [(xml_node, json_template)]

    while node_stack:
        xml_node, json_node = node_stack.pop()

        for xml_child in xml_node.iterchildren():
            key = xml_child.attrib.get(key_attrib)
            value = deepcopy(XML_DTYPE_TO_VALUE[xml_child.tag])

            if isinstance(json_node, dict):
                json_node[key] = value
            elif isinstance(json_node, list) and isinstance(value, dict):
                json_node.append(value)

            if isinstance(value, (dict, list)):
                node_stack.append((xml_child, value))

    return json_template
