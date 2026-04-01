---
name: bfl-api
description: BFL FLUX API integration guide covering endpoints, async polling patterns, rate limiting, error handling, webhooks, and regional endpoints with Python and TypeScript code examples.
metadata:
  author: Black Forest Labs
  version: "1.0.0"
  tags: flux, bfl, api, integration, webhooks, rate-limiting
---

# BFL API Integration Guide

Use this skill when integrating BFL FLUX APIs into applications for image generation, editing, and processing.

## First: Check API Key

```bash
echo $BFL_API_KEY
```

If empty or you see "Not authenticated" errors, get a key at https://dashboard.bfl.ai/get-started

## Important: Image URLs Expire in 10 Minutes

Result URLs from the API are temporary. Download images immediately after generation completes.

## Quick Reference

### Base Endpoints

| Region | Endpoint                | Use Case                    |
| ------ | ----------------------- | --------------------------- |
| Global | `https://api.bfl.ai`    | Default, automatic failover |
| EU     | `https://api.eu.bfl.ai` | GDPR compliance             |
| US     | `https://api.us.bfl.ai` | US data residency           |

### Model Endpoints & Pricing

> 1 credit = $0.01 USD. FLUX.2 uses megapixel-based pricing.

#### FLUX.2 Models

| Model             | Path                  | 1MP T2I | 1MP I2I | Best For                        |
| ----------------- | --------------------- | ------- | ------- | ------------------------------- |
| FLUX.2 [klein] 4B | `/v1/flux-2-klein-4b` | $0.014  | $0.015  | Real-time, high volume          |
| FLUX.2 [klein] 9B | `/v1/flux-2-klein-9b` | $0.015  | $0.017  | Balanced quality/speed          |
| FLUX.2 [pro]      | `/v1/flux-2-pro`      | $0.03   | $0.045  | Production, fast turnaround     |
| FLUX.2 [max]      | `/v1/flux-2-max`      | $0.07   | $0.10   | Maximum quality                 |
| FLUX.2 [flex]     | `/v1/flux-2-flex`     | $0.05   | $0.10   | Typography, adjustable controls |

### Authentication

```bash
x-key: YOUR_API_KEY
```

### Basic Request Flow

```
1. POST request to model endpoint
   -> Response: { "polling_url": "..." }

2. GET polling_url (repeat until complete)
   -> Response: { "status": "Pending" | "Ready" | "Error", ... }

3. When Ready, download result URL (expires in 10 minutes)
```

### Rate Limits

Standard: 24 concurrent requests.

### Quick Start

```bash
# Generate
curl -s -X POST "https://api.bfl.ai/v1/flux-2-pro" \
  -H "x-key: $BFL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A serene mountain landscape at sunset", "width": 1024, "height": 1024}'

# Poll (use polling_url from response)
curl -s "POLLING_URL" -H "x-key: $BFL_API_KEY"

# Download when Ready
curl -s -o output.png "IMAGE_URL"
```
