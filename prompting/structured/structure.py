from io import StringIO, BytesIO

import lxml.etree as ET
import guardrails as gd


wrapper_template = """
<rail version="0.1">

<output>
  <object name="{name}" description="{description}">
  </object>
  <string name="{name}_status" description="Status of the structure {name} completed or not." />
</output>

<prompt>
Extract the output from the description in the provided output schema.
</prompt>

</rail>
"""

NAME = "extraction"
DESCRIPTION = (
    "Details of the fields to be extracted from the vulnerability description."
)

output_wrapper = wrapper_template.format(name=NAME, description=DESCRIPTION)

wrapped_xml = ET.parse(StringIO(output_wrapper))
structure_xml = ET.parse("structure.xml")

obj_root = wrapped_xml.getroot().find("output/object")
for elem in structure_xml.getroot().iterchildren():
    obj_root.append(elem)

output = BytesIO()
wrapped_xml.write(output)
output = output.getvalue().decode("UTF-8")

rail = gd.Rail.from_string(output)
guard = gd.Guard.from_rail_string(output)
