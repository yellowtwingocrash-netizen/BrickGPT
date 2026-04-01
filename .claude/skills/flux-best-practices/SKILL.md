---
name: flux-best-practices
description: Comprehensive guide for BFL FLUX image generation models. Covers prompting, T2I, I2I, structured JSON, hex colors, typography, multi-reference editing, and model-specific best practices for FLUX.2 and FLUX.1 families.
metadata:
  author: Black Forest Labs
  version: "1.0.0"
  tags: flux, bfl, image-generation, prompting, t2i, i2i
---

# FLUX Best Practices

Use this skill when generating prompts for any BFL FLUX model to ensure optimal image quality and accurate prompt interpretation.

## When to Use

- Creating prompts for FLUX.2 or FLUX.1 models
- Text-to-image (T2I) generation
- Image-to-image (I2I) editing with FLUX.2 models
- Structured scene generation with JSON
- Typography and text rendering
- Multi-reference style transfer
- Color-accurate brand generations

## Quick Reference

### Prompt Structure Formula

```
[Subject] + [Action/Pose] + [Style/Medium] + [Context/Setting] + [Lighting] + [Camera/Technical]
```

### Model Selection

| Use Case            | Recommended Model | Notes                                  |
| ------------------- | ----------------- | -------------------------------------- |
| Fastest generation  | FLUX.2 [klein]    | 4B or 9B, sub-second                   |
| Highest quality     | FLUX.2 [max]      | Best detail, grounding search          |
| Production balanced | FLUX.2 [pro]      | Quality + speed                        |
| Typography/text     | FLUX.2 [flex]     | Best text rendering                    |
| Local/development   | FLUX.2 [dev]      | Open weights                           |
| Image editing       | FLUX.2 [pro/max]  | Pass image URL directly to input_image |
| Inpainting          | FLUX.1 Fill       | Object removal/completion              |
| Context editing     | FLUX.1 Kontext    | Older model, prefer FLUX.2             |

### Critical Rules

1. **NO negative prompts** - FLUX does not support negative prompts; describe what you want
2. **Be specific** - Vague prompts produce mediocre results
3. **Use natural language** - Prose/narrative style works best
4. **Specify lighting** - Lighting has the biggest impact on quality
5. **Quote text** - Use "quoted text" for typography rendering
6. **Hex colors** - Use #RRGGBB format with color description

## Example Prompt

```
A weathered fisherman in his 70s with deep wrinkles and a salt-and-pepper beard,
wearing a navy cable-knit sweater, standing at the helm of his wooden boat.
Golden hour sunlight from the left creates dramatic rim lighting on his profile.
Shot on Hasselblad with 85mm lens at f/2.8, shallow depth of field with harbor
lights creating soft bokeh in the background. Kodak Portra 400 color science.
```
