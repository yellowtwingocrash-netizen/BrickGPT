"""
Decompose a substation image or text description into component specs for BrickGPT.

Supports two modes:
1. Image decomposition via Claude/Gemini VLM
2. Text decomposition via Claude LLM
"""

import base64
import json
import os
from pathlib import Path

from .models import ComponentSpec

DECOMPOSITION_PROMPT = """\
You are an expert at decomposing complex structures into simple geometric components \
that can be built with LEGO bricks.

Analyze the input and decompose it into individual buildable components. \
For each component, provide a JSON object with:
- "name": short identifier (e.g., "control_building", "transformer_1")
- "description": 8-16 word description using geometric language suitable for LEGO generation. \
  Focus on shape: rectangular, flat, tall, compact, boxy, stepped, narrow, wide. \
  Mention structural features: legs, base, roof, columns, tiers, walls.
- "offset": [x, y, z] position offset in grid units (each unit = 1 LEGO stud). \
  Place ground-level components at z=0.
- "world_dim": estimated grid size needed (10-20, default 20)

The LEGO model uses a grid where each component occupies up to world_dim×world_dim×world_dim studs. \
Components will be placed in a larger grid at the specified offsets.

Categories the model handles well: buildings, towers, vehicles, furniture-like structures, \
boxy/rectangular shapes, walls, columns.

Return ONLY a JSON array of component objects. No other text."""

IMAGE_DECOMPOSITION_PROMPT = """\
Look at this image and decompose the structure into individual buildable LEGO components.

""" + DECOMPOSITION_PROMPT

TEXT_DECOMPOSITION_PROMPT = """\
Decompose the following structure description into individual buildable LEGO components:

"{description}"

""" + DECOMPOSITION_PROMPT


def decompose_image(
    image_path: str,
    api_key: str | None = None,
    model: str = 'claude-sonnet-4-20250514',
) -> list[ComponentSpec]:
    """Decompose an image into component specs using Claude VLM."""
    try:
        import anthropic
    except ImportError:
        raise ImportError('pip install anthropic  # required for VLM decomposition')

    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError('ANTHROPIC_API_KEY not set. Pass api_key or set the environment variable.')

    client = anthropic.Anthropic(api_key=api_key)

    # Read and encode image
    image_data = Path(image_path).read_bytes()
    base64_image = base64.standard_b64encode(image_data).decode('utf-8')

    suffix = Path(image_path).suffix.lower()
    media_type = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }.get(suffix, 'image/png')

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': media_type,
                        'data': base64_image,
                    },
                },
                {
                    'type': 'text',
                    'text': IMAGE_DECOMPOSITION_PROMPT,
                },
            ],
        }],
    )

    return _parse_components(response.content[0].text)


def decompose_text(
    description: str,
    api_key: str | None = None,
    model: str = 'claude-sonnet-4-20250514',
) -> list[ComponentSpec]:
    """Decompose a text description into component specs using Claude LLM."""
    try:
        import anthropic
    except ImportError:
        raise ImportError('pip install anthropic  # required for VLM decomposition')

    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError('ANTHROPIC_API_KEY not set. Pass api_key or set the environment variable.')

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            'role': 'user',
            'content': TEXT_DECOMPOSITION_PROMPT.format(description=description),
        }],
    )

    return _parse_components(response.content[0].text)


def _parse_components(response_text: str) -> list[ComponentSpec]:
    """Parse VLM JSON response into ComponentSpec list."""
    # Extract JSON array from response (handle markdown code blocks)
    text = response_text.strip()
    if '```' in text:
        text = text.split('```')[1]
        if text.startswith('json'):
            text = text[4:]
        text = text.strip()

    components_data = json.loads(text)

    return [
        ComponentSpec(
            name=c['name'],
            description=c['description'],
            offset=tuple(c.get('offset', [0, 0, 0])),
            world_dim=c.get('world_dim', 20),
        )
        for c in components_data
    ]
