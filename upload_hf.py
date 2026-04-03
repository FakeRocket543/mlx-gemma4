#!/usr/bin/env python3
"""Generate README for each model and upload all 12 variants to HuggingFace."""
import os, json
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
OWNER = "FakeRockert543"
BASE = "/Users/fl/Python/mlx_gemma4"

MODELS = {
    "E2B":     {"params": "2.3B", "audio": True,  "pipeline": "any-to-any",          "base": "google/gemma-4-E2B-it"},
    "E4B":     {"params": "4.5B", "audio": True,  "pipeline": "any-to-any",          "base": "google/gemma-4-E4B-it"},
    "26B-A4B": {"params": "26B MoE", "audio": False, "pipeline": "image-text-to-text", "base": "google/gemma-4-26B-A4B-it"},
    "31B":     {"params": "31B dense", "audio": False, "pipeline": "image-text-to-text", "base": "google/gemma-4-31B-it"},
}

SIZES = {
    "E2B":     {"4bit": "7.1 GB",  "8bit": "8.5 GB",  "bf16": "9.6 GB"},
    "E4B":     {"4bit": "10.3 GB", "8bit": "12.3 GB", "bf16": "16.0 GB"},
    "26B-A4B": {"4bit": "16.4 GB", "8bit": "28.6 GB", "bf16": "51.6 GB"},
    "31B":     {"4bit": "20.4 GB", "8bit": "35.1 GB", "bf16": "62.5 GB"},
}

PRECISIONS = ["4bit", "8bit", "bf16"]

def hf_repo_name(variant, prec):
    v = variant.lower()
    return f"gemma-4-{v}-it-MLX-{prec}"

def dir_name(variant, prec):
    v = variant.lower()
    tag = prec if prec == "bf16" else prec
    return f"gemma4-{v}-mlx-{tag}"

def sibling_links(variant, prec):
    lines = []
    for p in PRECISIONS:
        if p == prec:
            lines.append(f"- **{p}** ← you are here")
        else:
            lines.append(f"- [{p}](https://huggingface.co/{OWNER}/{hf_repo_name(variant, p)})")
    return "\n".join(lines)

def all_variants_table():
    rows = []
    for v in MODELS:
        for p in PRECISIONS:
            name = hf_repo_name(v, p)
            size = SIZES[v][p]
            rows.append(f"| [{name}](https://huggingface.co/{OWNER}/{name}) | {MODELS[v]['params']} | {p} | {size} | {'✅' if MODELS[v]['audio'] else '—'} |")
    return "\n".join(rows)

def gen_readme(variant, prec):
    m = MODELS[variant]
    size = SIZES[variant][prec]
    repo_name = hf_repo_name(variant, prec)
    is_quant = prec != "bf16"
    bits_str = prec.replace("bit", "") if is_quant else "—"

    audio_tags = """
- audio
- speech""" if m["audio"] else ""

    quant_note = ""
    if is_quant:
        quant_note = f"""
## Quantization Details

- **Bits:** {bits_str}
- **Group size:** 64
- **Mode:** affine
- **Strategy:** PLE-safe — only large `nn.Linear` and `SwitchLinear` (MoE) layers are quantized. All PLE/ScaledLinear/vision/audio layers stay in bf16."""
    else:
        quant_note = """
## Precision

Full bf16 weights, no quantization applied."""

    audio_usage = ""
    if m["audio"]:
        audio_usage = f"""
### Audio

```python
messages = [{{"role": "user", "content": [
    {{"type": "audio", "url": "speech.wav"}},
    {{"type": "text", "text": "What is the speaker saying? What is their emotional tone?"}},
]}}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, audio=["speech.wav"],
    max_tokens=200, repetition_penalty=1.2, temperature=0.1)
print(out.text)
```
"""

    return f"""---
language:
  - en
  - zh
  - ja
  - ko
  - de
  - fr
  - es
  - pt
  - it
  - ar
  - hi
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
library_name: mlx
pipeline_tag: {m["pipeline"]}
base_model: {m["base"]}
tags:
- mlx
- gemma4
- ple-safe
- quantized
- apple-silicon
- vision{audio_tags}
---

# {repo_name}

**PLE-safe** MLX {prec} weights for Google Gemma 4 {variant} ({m["params"]}) on Apple Silicon.

- 📦 Source & convert scripts: [GitHub — FakeRocket543/mlx-gemma4](https://github.com/FakeRocket543/mlx-gemma4)
- 📊 Size: **{size}**

> ⚠️ **Existing MLX quantized Gemma 4 models (mlx-community, unsloth) produce garbage output** due to quantizing PLE (Per-Layer Embedding) layers. This repo provides working quantized weights. See [Why](#why-ple-safe) below.

## Other Precisions

{sibling_links(variant, prec)}

## All Gemma 4 MLX Models

| Model | Params | Precision | Size | Audio |
|---|---|---|---|---|
{all_variants_table()}
{quant_note}

## Why PLE-Safe?

Gemma 4 uses a novel **PLE (Per-Layer Embeddings)** architecture with `ScaledLinear` layers that multiply outputs by a learned scalar. Standard quantization introduces rounding error in these layers, and the scalar amplifies it — producing `ionoxffionoxff...` garbage.

**Our fix:** Only quantize the large decoder `nn.Linear` and `SwitchLinear` (MoE expert) layers. Everything else stays bf16:

| Quantized ({prec}) | Kept in bf16 |
|---|---|
| Attention projections (q/k/v/o_proj) | ScaledEmbedding (embed_tokens) |
| MLP layers (gate/up/down_proj) | ScaledLinear (PLE pathway) |
| MoE expert layers (SwitchLinear) | Per-layer embeddings (per_layer_*) |
| | Vision encoder |
{"| | Audio encoder |" if m["audio"] else ""}| | All norms and scalars |

## Usage

**Prerequisite:** Apply the ScaledLinear fix to mlx-vlm (required until PR merged upstream):

```bash
pip install mlx-vlm

# Apply fix
git clone https://github.com/FakeRocket543/mlx-gemma4.git
cp mlx-gemma4/mlx_vlm_patches/models/gemma4/language.py \\
   $(python -c "import mlx_vlm; print(mlx_vlm.__path__[0])")/models/gemma4/
```

**Important:** You must manually apply the chat template. `mlx_vlm.generate()` does not do this automatically for Gemma 4.

### Vision

```python
from mlx_vlm import load, generate

model, processor = load("{OWNER}/{repo_name}")
tokenizer = processor.tokenizer

messages = [{{"role": "user", "content": [
    {{"type": "image", "url": "photo.jpg"}},
    {{"type": "text", "text": "Describe this image in detail."}},
]}}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, ["photo.jpg"],
    max_tokens=200, repetition_penalty=1.2, temperature=0.7)
print(out.text)
```
{audio_usage}
### Text

```python
messages = [{{"role": "user", "content": "What is the capital of France?"}}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, max_tokens=100, temperature=0.0)
print(out.text)
```

## Bugs Fixed in mlx-vlm

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | `ScaledLinear` inherits `nn.Module` not `nn.Linear` | `nn.quantize()` can't find these layers | Change to `ScaledLinear(nn.Linear)` |
| 2 | Standard quantization quantizes PLE layers | Garbage output on 4-bit/8-bit | PLE-safe `class_predicate` skipping PLE/vision/audio |
| 3 | `processor.save_pretrained()` strips `feature_extractor` | Audio silently dropped | Copy `processor_config.json` from source |
| 4 | `SwitchLinear` (MoE) not quantized | 26B-A4B: 49 GB instead of 16 GB | Check `hasattr(module, 'to_quantized')` |

Fixed source files are included in the [GitHub repo](https://github.com/FakeRocket543/mlx-gemma4/tree/main/mlx_vlm_patches).

## Convert From Source

```bash
git clone https://github.com/FakeRocket543/mlx-gemma4.git
cd mlx-gemma4
python convert_gemma4.py {variant} {bits_str if is_quant else "bf16"}
```

## Validation

All 12 variants validated on 10 images + {"12 audio samples + " if m["audio"] else ""}3 chat prompts. Full results: [GitHub](https://github.com/FakeRocket543/mlx-gemma4).

## License

Model weights: [Google Gemma License](https://ai.google.dev/gemma/terms). Scripts: MIT.
"""

if __name__ == "__main__":
    api = HfApi(token=TOKEN)

    for variant in MODELS:
        for prec in PRECISIONS:
            repo_name = hf_repo_name(variant, prec)
            full_repo = f"{OWNER}/{repo_name}"
            local_dir = os.path.join(BASE, dir_name(variant, prec))

            if not os.path.isdir(local_dir):
                print(f"SKIP {local_dir} (not found)")
                continue

            # Generate README
            readme = gen_readme(variant, prec)
            readme_path = os.path.join(local_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(readme)
            print(f"✓ Generated README for {repo_name}")

            # Create repo
            api.create_repo(full_repo, repo_type="model", exist_ok=True)
            print(f"✓ Repo {full_repo} ready")

            # Upload entire folder
            print(f"  Uploading {local_dir} ...")
            api.upload_folder(
                folder_path=local_dir,
                repo_id=full_repo,
                commit_message=f"Upload {repo_name} — PLE-safe MLX weights",
                ignore_patterns=[".*"],
            )
            print(f"✓ Uploaded {full_repo}\n")
