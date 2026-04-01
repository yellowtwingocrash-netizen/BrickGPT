"""
Generate individual LEGO components using BrickGPT.

Each ComponentSpec is generated independently in its own grid,
then returned alongside its spec for later composition.
"""

from dataclasses import dataclass

from brickgpt.data import BrickStructure
from brickgpt.models.brickgpt import BrickGPT, BrickGPTConfig

from .models import ComponentSpec, PipelineConfig


@dataclass
class ComponentResult:
    """Result of generating a single component."""
    spec: ComponentSpec
    bricks: BrickStructure
    n_bricks: int
    n_regenerations: int
    rejection_reasons: dict


def generate_components(
    specs: list[ComponentSpec],
    config: PipelineConfig | None = None,
) -> list[ComponentResult]:
    """
    Generate each component as a separate BrickStructure using BrickGPT.

    Each component gets its own BrickGPT instance configured for its grid size.
    """
    if config is None:
        config = PipelineConfig()

    results = []
    for i, spec in enumerate(specs):
        print(f'[{i + 1}/{len(specs)}] Generating: {spec.name} — "{spec.description}"')

        cfg = BrickGPTConfig(
            model_name_or_path=config.model_name_or_path,
            world_dim=spec.world_dim,
            max_bricks=config.max_bricks_per_component,
            max_regenerations=config.max_regenerations,
            use_gurobi=config.use_gurobi,
            temperature=spec.temperature,
        )

        model = BrickGPT(cfg)

        if spec.seed is not None:
            import torch
            torch.manual_seed(spec.seed)

        result = model(spec.description)

        component_result = ComponentResult(
            spec=spec,
            bricks=result['bricks'],
            n_bricks=len(result['bricks']),
            n_regenerations=result['n_regenerations'],
            rejection_reasons=dict(result['rejection_reasons']),
        )
        results.append(component_result)

        print(f'  → {component_result.n_bricks} bricks, '
              f'{component_result.n_regenerations} regenerations')

    return results
