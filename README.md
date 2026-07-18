# CLIP Encoding - ManimGL Explainer

ManimGL animation source code for explaining how CLIP maps images and text into a shared embedding space.

This repo is a visual explainer for the core CLIP idea: image patches and text tokens are encoded separately, projected into the same vector space, normalized, compared with cosine similarity, and then used for retrieval or classification-like decisions.

## What This Teaches

- How image and text encoders produce vectors.
- Why a shared embedding space makes image-text matching possible.
- How L2 normalization and cosine similarity shape CLIP comparisons.
- How similarity matrices, UMAP-style projections, and decision boundaries can make embedding behavior easier to see.

## Repo Map

- `clip_encoding.py` - main CLIP encoding overview scene.
- `clip_encoding_scene_1.py` - shared embedding space.
- `clip_encoding_scene_2.py` - CLIP similarity matrix.
- `clip_encoding_scene_3.py` - L2 normalization and cosine similarity.
- `clip_encoding_scene_4.py` - UMAP-style embedding visualization.
- `clip_encoding_scene_5.py` - MLP decision boundary scene.
- `clip_encoding_scene_6.py` - bad sinks and cluster behavior.
- `example_photos/` - small reference images used by the scenes.
- `requirements.txt` - pinned ManimGL-related dependencies.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the PyPI `manimgl` package has import issues on your Python version, install ManimGL from source instead:

```bash
pip install /path/to/manimgl
```

## Running A Scene

Interactive mode with live preview:

```bash
manimgl clip_encoding.py CLIPEncoding -i --autoreload
```

Render the main scene to video:

```bash
manimgl clip_encoding.py CLIPEncoding -w --hd
```

Render a specific scene:

```bash
manimgl clip_encoding_scene_3.py L2NormalizationCosineSimilarity -w --hd
```

Rendered output is written under `videos/`.

## Why This Exists

CLIP is often introduced with a single diagram, but the intuition lives in the geometry: normalization, similarity, clusters, and boundaries. This repo keeps the animation source public so the explanation is inspectable and reusable.

## Related Explainers

- [LLM Arithmetic](https://github.com/leannchen86/llm-arithmetic-manim)
- [Hybrid Search + BM25](https://github.com/leannchen86/hybrid-search-bm25-manim)
- [Neural Superposition](https://github.com/leannchen86/neural-superposition-manim)
