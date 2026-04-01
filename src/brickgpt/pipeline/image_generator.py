"""
Generate photorealistic reference images using BFL FLUX API.

These images serve as input to the VLM decomposer for
identifying substation components and their spatial relationships.

Uses the BFL API with proper polling pattern per the bfl-api skill.
Supports FLUX.2 [pro] by default (best balance of quality and speed).
"""

import os
import time
from pathlib import Path


def generate_reference_image(
    prompt: str,
    output_path: str,
    api_key: str | None = None,
    model: str = 'flux-2-pro',
    base_url: str = 'https://api.bfl.ai',
    width: int = 1024,
    height: int = 1024,
) -> str:
    """
    Generate a photorealistic reference image using BFL FLUX API.

    Uses async polling pattern: POST \u2192 get polling_url \u2192 poll until Ready \u2192 download.
    Result URLs expire in 10 minutes \u2014 downloads immediately.

    Returns the path to the saved image.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError('pip install httpx  # required for Flux image generation')

    api_key = api_key or os.environ.get('BFL_API_KEY')
    if not api_key:
        raise ValueError('BFL_API_KEY not set. Pass api_key or set the environment variable.')

    headers = {'x-key': api_key, 'Content-Type': 'application/json'}
    client = httpx.Client(timeout=60.0, headers=headers)

    # Step 1: Submit generation request
    endpoint = f'{base_url}/v1/{model}'
    response = client.post(
        endpoint,
        json={
            'prompt': prompt,
            'width': width,
            'height': height,
        },
    )
    response.raise_for_status()
    data = response.json()
    polling_url = data.get('polling_url')
    if not polling_url:
        # Fallback for older API format
        request_id = data['id']
        polling_url = f'{base_url}/v1/get_result?id={request_id}'

    # Step 2: Poll for result with exponential backoff
    delay = 1.0
    for _ in range(120):  # Max ~5 minutes with backoff
        result = client.get(polling_url)
        result.raise_for_status()
        result_data = result.json()

        status = result_data.get('status')
        if status == 'Ready':
            image_url = result_data['result']['sample']
            # Step 3: Download immediately (URL expires in 10 minutes)
            img_response = client.get(image_url)
            img_response.raise_for_status()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(img_response.content)
            return output_path
        elif status == 'Error':
            error = result_data.get('error', 'Unknown generation error')
            raise RuntimeError(f'FLUX generation failed: {error}')

        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)  # Exponential backoff, cap at 5s

    raise TimeoutError('FLUX image generation timed out')


def generate_substation_views(
    description: str,
    output_dir: str,
    api_key: str | None = None,
    model: str = 'flux-2-pro',
) -> list[str]:
    """
    Generate multiple reference views of a substation for better VLM decomposition.

    Returns list of image paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    views = [
        {
            'name': 'aerial',
            'suffix': ', aerial view, 3/4 angle from above, clear daylight, photorealistic industrial photography',
        },
        {
            'name': 'front',
            'suffix': ', front elevation view, eye level, photorealistic architectural photography',
        },
    ]

    paths = []
    for view in views:
        prompt = f'{description}{view["suffix"]}'
        path = str(out / f'reference_{view["name"]}.png')
        print(f'Generating {view["name"]} view...')
        generate_reference_image(prompt, path, api_key=api_key, model=model)
        paths.append(path)
        print(f'  \u2192 Saved to {path}')

    return paths
