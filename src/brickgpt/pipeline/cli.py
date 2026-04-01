"""
CLI entry point for the compositional Lego generation pipeline.

Usage:
    # From text description (uses Claude to decompose into components)
    uv run generate-substation --prompt "Large electrical onshore substation"

    # From reference image (uses Claude VLM to decompose)
    uv run generate-substation --image substation.png

    # From pre-defined component specs
    uv run generate-substation --components components.json

    # Full pipeline with Flux2.pro image generation
    uv run generate-substation --prompt "..." --generate-image
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Generate a large Lego structure from a compositional pipeline.',
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--prompt', type=str,
                             help='Text description of the structure to generate')
    input_group.add_argument('--image', type=str,
                             help='Path to a reference image for VLM decomposition')
    input_group.add_argument('--components', type=str,
                             help='Path to a JSON file with pre-defined component specs')

    parser.add_argument('--generate-image', action='store_true',
                        help='Generate reference images via Flux2.pro before decomposition')
    parser.add_argument('--output-dir', type=str, default='./substation_output',
                        help='Output directory (default: ./substation_output)')
    parser.add_argument('--model', type=str, default='AvaLovelace/BrickGPT',
                        help='BrickGPT model path')
    parser.add_argument('--world-dim', type=int, default=80,
                        help='Final composed grid dimension (default: 80)')
    parser.add_argument('--max-bricks', type=int, default=500,
                        help='Max bricks per component (default: 500)')
    parser.add_argument('--no-gurobi', action='store_true',
                        help='Disable Gurobi stability analysis')
    parser.add_argument('--vlm-model', type=str, default='claude-sonnet-4-20250514',
                        help='Claude model for VLM decomposition')

    args = parser.parse_args()

    from .models import PipelineConfig, ComponentSpec
    from .instructions import save_outputs

    config = PipelineConfig(
        model_name_or_path=args.model,
        max_bricks_per_component=args.max_bricks,
        final_world_dim=args.world_dim,
        use_gurobi=not args.no_gurobi,
        anthropic_model=args.vlm_model,
    )

    # Step 1: Get component specs
    if args.components:
        # Load pre-defined specs
        specs_data = json.loads(Path(args.components).read_text())
        specs = [ComponentSpec.from_dict(d) for d in specs_data]
        print(f'Loaded {len(specs)} component specs from {args.components}')

    elif args.image:
        # Decompose from image
        from .decomposer import decompose_image
        print(f'Decomposing image: {args.image}')
        specs = decompose_image(args.image, model=config.anthropic_model)
        print(f'Identified {len(specs)} components')

    elif args.prompt:
        if args.generate_image:
            # Generate reference images first
            from .image_generator import generate_substation_views
            print(f'Generating reference images with Flux2.pro...')
            image_paths = generate_substation_views(
                args.prompt, output_dir=f'{args.output_dir}/reference_images',
            )
            # Decompose from generated image
            from .decomposer import decompose_image
            print(f'Decomposing generated image...')
            specs = decompose_image(image_paths[0], model=config.anthropic_model)
        else:
            # Text-only decomposition
            from .decomposer import decompose_text
            print(f'Decomposing text description...')
            specs = decompose_text(args.prompt, model=config.anthropic_model)

        print(f'Identified {len(specs)} components:')
        for s in specs:
            print(f'  - {s.name}: "{s.description}" @ offset {s.offset}')

    # Step 2: Generate each component
    from .component_generator import generate_components
    print(f'\nGenerating {len(specs)} components with BrickGPT...')
    results = generate_components(specs, config)

    # Step 3: Compose
    from .composer import compose_structures
    print(f'\nComposing into {config.final_world_dim}x{config.final_world_dim}x{config.final_world_dim} grid...')
    composed = compose_structures(results, world_dim=config.final_world_dim)

    # Step 4: Save outputs
    print(f'\nSaving outputs to {args.output_dir}/')
    outputs = save_outputs(results, composed, args.output_dir)

    print(f'\nDone! Output files:')
    for key, path in outputs.items():
        print(f'  {key}: {path}')

    # Print BOM summary
    bom = composed.bill_of_materials()
    total_bricks = sum(v['count'] for v in bom.values())
    print(f'\nBill of Materials ({total_bricks} bricks total):')
    for brick_type, info in bom.items():
        print(f'  {brick_type}: {info["count"]}x (LEGO Part #{info["part_id"].replace(".DAT", "")})')


if __name__ == '__main__':
    main()
