"""
Generate building instructions and bill of materials from a composed BrickStructure.
"""

import json
from pathlib import Path

from brickgpt.data import BrickStructure

from .component_generator import ComponentResult


def generate_bom(structure: BrickStructure) -> dict[str, dict]:
    """
    Generate a bill of materials from a BrickStructure.
    Returns a dict mapping brick notation (e.g., '2x4') to count and part ID.
    """
    return structure.bill_of_materials()


def generate_ldr_instructions(
    results: list[ComponentResult],
    composed: BrickStructure,
) -> str:
    """
    Generate an LDraw file with component-grouped building steps.

    Each component gets a group header comment and its bricks are
    separated by STEP markers, making building instructions follow
    a component-by-component assembly order.
    """
    lines = []
    lines.append('0 FILE substation.ldr\n')
    lines.append('0 BrickGPT Compositional Pipeline Output\n')
    lines.append('0 STEP\n')

    for result in results:
        ox, oy, oz = result.spec.offset
        lines.append(f'0 // Component: {result.spec.name}\n')
        lines.append(f'0 // Description: {result.spec.description}\n')

        for brick in result.bricks:
            from brickgpt.data.brick_structure import Brick
            shifted = Brick(
                h=brick.h, w=brick.w,
                x=brick.x + ox, y=brick.y + oy, z=brick.z + oz,
            )
            lines.append(shifted.to_ldr())

        lines.append('0 STEP\n')

    return ''.join(lines)


def save_outputs(
    results: list[ComponentResult],
    composed: BrickStructure,
    output_dir: str,
) -> dict[str, str]:
    """
    Save all output files to the specified directory.

    Returns a dict of output file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Composed structure as txt
    txt_path = out / 'substation.txt'
    txt_path.write_text(composed.to_txt())
    outputs['txt'] = str(txt_path)

    # Composed structure as LDR with component grouping
    ldr_path = out / 'substation.ldr'
    ldr_content = generate_ldr_instructions(results, composed)
    ldr_path.write_text(ldr_content)
    outputs['ldr'] = str(ldr_path)

    # Bill of materials
    bom = generate_bom(composed)
    bom_path = out / 'bill_of_materials.json'
    bom_path.write_text(json.dumps(bom, indent=2))
    outputs['bom'] = str(bom_path)

    # Component specs (for reproducibility)
    specs_path = out / 'components.json'
    specs_data = [r.spec.to_dict() for r in results]
    specs_path.write_text(json.dumps(specs_data, indent=2))
    outputs['specs'] = str(specs_path)

    # Per-component LDR files
    components_dir = out / 'components'
    components_dir.mkdir(exist_ok=True)
    for result in results:
        comp_path = components_dir / f'{result.spec.name}.ldr'
        comp_path.write_text(result.bricks.to_ldr())
        comp_path_txt = components_dir / f'{result.spec.name}.txt'
        comp_path_txt.write_text(result.bricks.to_txt())

    outputs['components_dir'] = str(components_dir)

    return outputs
