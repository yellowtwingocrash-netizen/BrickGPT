"""Data models for the compositional generation pipeline."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ComponentSpec:
    """Specification for a single substation component to be generated."""
    name: str
    description: str  # BrickGPT-friendly text prompt (8-16 words)
    offset: tuple[int, int, int] = (0, 0, 0)  # (x, y, z) offset in final grid
    world_dim: int = 20  # Per-component generation grid size
    temperature: float = 0.6
    seed: int | None = None

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'offset': list(self.offset),
            'world_dim': self.world_dim,
            'temperature': self.temperature,
            'seed': self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ComponentSpec':
        return cls(
            name=d['name'],
            description=d['description'],
            offset=tuple(d.get('offset', (0, 0, 0))),
            world_dim=d.get('world_dim', 20),
            temperature=d.get('temperature', 0.6),
            seed=d.get('seed'),
        )


@dataclass
class PipelineConfig:
    """Configuration for the full generation pipeline."""
    # Generation engine
    model_name_or_path: str = 'AvaLovelace/BrickGPT'
    max_bricks_per_component: int = 500
    max_regenerations: int = 50
    use_gurobi: bool = True

    # Composition
    final_world_dim: int = 80  # Size of the composed grid

    # VLM decomposition
    vlm: Literal['claude', 'gemini'] = 'claude'
    anthropic_api_key: str | None = None
    anthropic_model: str = 'claude-sonnet-4-20250514'

    # Image generation
    flux_api_key: str | None = None
    flux_api_url: str = 'https://api.bfl.ml/v1/flux-pro-1.1'

    # Output
    output_dir: str = './output'
