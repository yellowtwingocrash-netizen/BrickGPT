"""
Compositional pipeline for generating large Lego structures from text/image descriptions.

Pipeline: Flux2.pro → VLM decomposition → BrickGPT per-component generation → composition
"""

from .models import ComponentSpec, PipelineConfig

__all__ = [
    'ComponentSpec',
    'PipelineConfig',
]
