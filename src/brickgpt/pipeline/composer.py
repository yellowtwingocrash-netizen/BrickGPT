"""
Compose multiple generated BrickStructure components into a single large structure.

Each component is placed at its specified offset in a larger grid.
"""

from brickgpt.data import BrickStructure

from .component_generator import ComponentResult


def compose_structures(
    results: list[ComponentResult],
    world_dim: int = 80,
) -> BrickStructure:
    """
    Merge all component BrickStructures into a single large structure.

    Each component's bricks are offset by the component's spec.offset.
    Bricks that collide or go out of bounds are skipped with a warning.
    """
    merged = BrickStructure([], world_dim=world_dim)
    total_skipped = 0

    for result in results:
        skipped = merged.merge(result.bricks, offset=result.spec.offset)
        if skipped:
            print(f'  Warning: {len(skipped)} bricks skipped in "{result.spec.name}" '
                  f'(collision or out of bounds)')
            total_skipped += len(skipped)

    print(f'Composed {len(merged)} bricks total'
          + (f' ({total_skipped} skipped)' if total_skipped else ''))

    return merged
