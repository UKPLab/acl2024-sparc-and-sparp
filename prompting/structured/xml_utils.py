from pathlib import Path

from lxml import etree


ENCLOSER_HOLDER = "extraction"
HOLDER_DESC = "Details of the fields to be extracted from the document."

ENCLOSER_ENDER = "extraction_status"
ENDER_DESC = "Status of the extraction completion. Always mark as `Done`."

XML_ENCLOSER_TEMPLATE = """
<output>
  <object name="{encloser_holder}" description="{holder_desc}">
  </object>
  <string name="{encloser_ender}" description="{ender_desc}" />
</output>
"""

XML_ENCLOSER = XML_ENCLOSER_TEMPLATE.format(
    encloser_holder=ENCLOSER_HOLDER,
    holder_desc=HOLDER_DESC,
    encloser_ender=ENCLOSER_ENDER,
    ender_desc=ENDER_DESC,
)

JSON_DTYPE_TO_XML_TAG = {
    str: "string",
    int: "integer",
    float: "float",
    list: "list",
    dict: "object",
}


def get_xml_object(xml_input):
    if isinstance(xml_input, str):
        if Path(xml_input).exists():
            xml_input = etree.parse(xml_input).getroot()
        else:
            xml_input = etree.fromstring(xml_input)
    return xml_input


def enclose_in_xml(
    xml_input,
    encloser_xml=XML_ENCLOSER,
    encloser_xpath="object",  # tag with path
    encloser_attrib_key="name",
    encloser_attrib_value=ENCLOSER_HOLDER,
    return_as_string=True,
):
    xml_input = get_xml_object(xml_input)

    encloser_xml = get_xml_object(encloser_xml)
    encloser_node = encloser_xml.findall(encloser_xpath)

    for node in encloser_node:
        if node.attrib.get(encloser_attrib_key) == encloser_attrib_value:
            encloser_node = node
            break

    for elem in xml_input.iterchildren():
        encloser_node.append(elem)

    etree.indent(encloser_xml, space="  ")
    if return_as_string:
        encloser_xml = etree.tostring(encloser_xml).decode("utf-8")

    return encloser_xml


def xml_template_from_json(
    json_sample,
    root_tag="output",
    key_attrib="name",
    placeholder_attribs=["description"],
    return_as_string=True,
):
    """Helper function to generate xml templates from a json sample in which the
    placeholder attributes such as descriptions can later be inserted."""

    if isinstance(placeholder_attribs, str):
        placeholder_attribs = [placeholder_attribs]
    attrib_dict = {i: "" for i in placeholder_attribs}

    xml_root = etree.fromstring(f"<{root_tag}></{root_tag}>")
    node_stack = [(xml_root, json_sample)]

    while node_stack:
        xml_node, json_node = node_stack.pop()

        for key, value in json_node.items():
            tag = JSON_DTYPE_TO_XML_TAG.get(type(value), "string")
            xml_child = etree.Element(tag, name=key, **attrib_dict)

            xml_node.append(xml_child)

            if isinstance(value, dict):
                node_stack.append((xml_child, value))

            elif isinstance(value, list) and value:
                tag = JSON_DTYPE_TO_XML_TAG.get(type(value[0]), "string")
                grand_child = etree.Element(tag)

                if isinstance(value[0], dict):
                    node_stack.append((grand_child, value[0]))
                xml_child.append(grand_child)

    etree.indent(xml_root, space="  ")
    if return_as_string:
        xml_root = etree.tostring(xml_root).decode("utf-8")

    return xml_root
