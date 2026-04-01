"""
Generate photorealistic reference images using Flux2.pro API.

These images serve as input to the VLM decomposer for
identifying substation components and their spatial relationships.
"""

import os
import time
from pathlib import Path


def generate_reference_image(
    prompt: str,
    output_path: str,
    api_key: str | None = None,
    api_url: str = 'https://api.bfl.ml/v1/flux-pro-1.1',
    width: int = 1024,
    height: int = 1024,
) -> str:
    """
    Generate a photorealistic reference image using Flux2.pro API.

    Returns the path to the saved image.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError('pip install httpx  # required for Flux2.pro image generation')

    api_key = api_key or os.environ.get('BFL_API_KEY')
    if not api_key:
        raise ValueError('BFL_API_KEY not set. Pass api_key or set the environment variable.')

    # Submit generation request
    client = httpx.Client(timeout=60.0)
    response = client.post(
        api_url,
        headers={'x-key': api_key},
        json={
            'prompt': prompt,
            'width': width,
            'height': height,
        },
    )
    response.raise_for_status()
    request_id = response.json()['id']

    # Poll for result
    result_url = 'https://api.bfl.ml/v1/get_result'
    for _ in range(120):  # Max 2 minutes
        result = client.get(result_url, params={'id': request_id})
        result.raise_for_status()
        data = result.json()
        if data['status'] == 'Ready':
            image_url = data['result']['sample']
            # Download and save
            img_response = client.get(image_url)
            img_response.raise_for_status()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(img_response.content)
            return output_path
        time.sleep(1)

    raise TimeoutError('Flux2.pro image generation timed out after 120 seconds')


def generate_substation_views(
    description: str,
    output_dir: str,
    api_key: str | None = None,
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
            'suffix': ', aerial view, 3/4 angle from above, clear daylight, photorealistic',
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
        generate_reference_image(prompt, path, api_key=api_key)
        paths.append(path)
        print(f'  \u2192 Saved to {path}')

    return paths
