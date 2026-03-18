from pydantic import BaseModel, Field


class ComponentDetail(BaseModel):
    """A single identified electrical/electronic component."""

    name: str = Field(description="Common component name, e.g. 'Resistor', 'NPN Transistor'.")
    type: str = Field(description="Specific variant or package, e.g. 'Carbon film, 5-band', 'TO-92 package'.")
    value: str | None = Field(
        default=None,
        description="Interpreted spec such as '10kΩ ±5%' or '100µF 25V'. None if not readable.",
    )
    markings: str | None = Field(
        default=None,
        description="Raw visible text from color bands, labels, or printed codes. None if none visible.",
    )
    function: str = Field(description="1-2 sentence description of what this component does in a circuit.")
    orientation: str | None = Field(
        default=None,
        description="Polarity or orientation notes. None if the component is non-polar.",
    )
    confidence: str = Field(description="One of: 'high', 'medium', 'low' — how clearly the component was identified.")


class IdentifyComponentResult(BaseModel):
    """Structured result for the identify_component intent."""

    components: list[ComponentDetail] = Field(description="Each identified component in the image.")
    summary: str = Field(description="Brief plain-English overview tying the components together.")
