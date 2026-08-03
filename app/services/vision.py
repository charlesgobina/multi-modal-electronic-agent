import asyncio
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.services.image_processing import validate_and_prepare_image
from app.services.model_client import get_llm


TASK_INSTRUCTIONS = {
    "describe_scene": (
        "The student wants to understand what is in front of them.\n"
        "1. Identify every distinct component, tool, and wire visible in the image.\n"
        "2. For each item, state its name, likely purpose in this setup, and how it relates to the other items.\n"
        "3. Describe the overall setup: what experiment or circuit is this likely part of?\n"
        "4. Call out anything that looks unusual, misplaced, or potentially problematic."
    ),
    "identify_component": (
        "The student needs help identifying one or more electrical/electronic components.\n"
        "Respond directly with the identified components and their functions.\n"
        "Do not start with scene-setting phrases like 'The image shows', 'I can see', or similar.\n"
        "Start with the first identified component immediately.\n"
        "For each component you can identify:\n"
        "- **Name & type**: e.g. '10kΩ resistor (carbon film, 5-band)', 'NPN transistor (TO-92 package)'.\n"
        "- **Key specs**: read color bands, markings, or labels if visible; state values and tolerances.\n"
        "- **Function**: what this component does in a circuit (1-2 sentences).\n"
        "- **Orientation/polarity**: if the component has polarity (LED, electrolytic cap, diode), "
        "state whether it appears to be oriented correctly based on visible connections.\n"
        "Use concise plain text or bullets, prioritizing the answer over narration.\n"
        "If you cannot read markings clearly, say so and give your best estimate with reasoning."
    ),
    "read_text": (
        "The student wants you to read text visible in the image (labels, datasheets, manuals, screens).\n"
        "1. Transcribe the relevant text accurately.\n"
        "2. Explain what the text means in context — e.g. what a datasheet parameter implies for their circuit.\n"
        "3. Highlight the most important information the student should pay attention to."
    ),
    "analyze_circuit": (
        "The student wants to know if their circuit connections are correct.\n"
        "Analyze step-by-step:\n"
        "1. **Power path**: trace VCC/GND connections. Are power and ground connected correctly? "
        "Any short circuits?\n"
        "2. **Signal path**: trace the signal flow from input to output. Are components connected "
        "in the right order and to the correct pins?\n"
        "3. **Component orientation**: check polarity of LEDs, diodes, electrolytic caps, and IC pin orientation.\n"
        "4. **Common mistakes**: look for floating pins, missing pull-up/pull-down resistors, "
        "incorrect resistor values for LEDs, backwards components, or bridged rows on breadboards.\n"
        "5. **Verdict**: clearly state whether the circuit looks correct or what specific changes are needed.\n\n"
        "If you cannot trace a connection clearly from the image, say so rather than guessing."
    ),
}


async def identify_image_streaming(image_bytes: bytes, prompt: str) -> AsyncIterator[str]:
    base64_image, media_type = await asyncio.to_thread(validate_and_prepare_image, image_bytes)
    settings = get_settings()
    llm = get_llm()

    messages = [
        SystemMessage(content=settings.system_prompt),
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}",
                    "detail": "auto",
                },
            },
            {"type": "text", "text": prompt or "What is in this image?"},
        ]),
    ]

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


async def identify_image(
    image_b64: str,
    media_type: str,
    prompt: str,
    task_type: str,
    memory_context: str,
) -> str:
    settings = get_settings()
    llm = get_llm()

    system_content = (
        f"{settings.system_prompt}\n"
        f"Current task mode: {task_type}.\n"
        f"Task instruction: {TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS['describe_scene'])}\n"
        f"Recent session context:\n{memory_context}"
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_b64}",
                    "detail": "auto",
                },
            },
            {"type": "text", "text": prompt or "What is in this image?"},
        ]),
    ]

    response = await llm.ainvoke(messages)
    text = response.content if isinstance(response.content, str) else ""
    return text.strip()
