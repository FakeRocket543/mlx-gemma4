---
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
pipeline_tag: any-to-any
base_model: google/gemma-4-E4B-it
tags:
- mlx
- gemma4
- ple-safe
- quantized
- apple-silicon
- vision
- audio
- speech
---

# gemma-4-e4b-it-MLX-8bit

**PLE-safe** MLX 8bit weights for Google Gemma 4 E4B (4.5B) on Apple Silicon.

- 📦 Source & convert scripts: [GitHub — FakeRocket543/mlx-gemma4](https://github.com/FakeRocket543/mlx-gemma4)
- 📊 Size: **12.3 GB**

> ⚠️ **Existing MLX quantized Gemma 4 models (mlx-community, unsloth) produce garbage output** due to quantizing PLE (Per-Layer Embedding) layers. This repo provides working quantized weights. See [Why](#why-ple-safe) below.

## Other Precisions

- [4bit](https://huggingface.co/FakeRockert543/gemma-4-e4b-it-MLX-4bit)
- **8bit** ← you are here
- [bf16](https://huggingface.co/FakeRockert543/gemma-4-e4b-it-MLX-bf16)

## All Gemma 4 MLX Models

| Model | Params | Precision | Size | Audio |
|---|---|---|---|---|
| [gemma-4-e2b-it-MLX-4bit](https://huggingface.co/FakeRockert543/gemma-4-e2b-it-MLX-4bit) | 2.3B | 4bit | 7.1 GB | ✅ |
| [gemma-4-e2b-it-MLX-8bit](https://huggingface.co/FakeRockert543/gemma-4-e2b-it-MLX-8bit) | 2.3B | 8bit | 8.5 GB | ✅ |
| [gemma-4-e2b-it-MLX-bf16](https://huggingface.co/FakeRockert543/gemma-4-e2b-it-MLX-bf16) | 2.3B | bf16 | 9.6 GB | ✅ |
| [gemma-4-e4b-it-MLX-4bit](https://huggingface.co/FakeRockert543/gemma-4-e4b-it-MLX-4bit) | 4.5B | 4bit | 10.3 GB | ✅ |
| [gemma-4-e4b-it-MLX-8bit](https://huggingface.co/FakeRockert543/gemma-4-e4b-it-MLX-8bit) | 4.5B | 8bit | 12.3 GB | ✅ |
| [gemma-4-e4b-it-MLX-bf16](https://huggingface.co/FakeRockert543/gemma-4-e4b-it-MLX-bf16) | 4.5B | bf16 | 16.0 GB | ✅ |
| [gemma-4-26b-a4b-it-MLX-4bit](https://huggingface.co/FakeRockert543/gemma-4-26b-a4b-it-MLX-4bit) | 26B MoE | 4bit | 16.4 GB | — |
| [gemma-4-26b-a4b-it-MLX-8bit](https://huggingface.co/FakeRockert543/gemma-4-26b-a4b-it-MLX-8bit) | 26B MoE | 8bit | 28.6 GB | — |
| [gemma-4-26b-a4b-it-MLX-bf16](https://huggingface.co/FakeRockert543/gemma-4-26b-a4b-it-MLX-bf16) | 26B MoE | bf16 | 51.6 GB | — |
| [gemma-4-31b-it-MLX-4bit](https://huggingface.co/FakeRockert543/gemma-4-31b-it-MLX-4bit) | 31B dense | 4bit | 20.4 GB | — |
| [gemma-4-31b-it-MLX-8bit](https://huggingface.co/FakeRockert543/gemma-4-31b-it-MLX-8bit) | 31B dense | 8bit | 35.1 GB | — |
| [gemma-4-31b-it-MLX-bf16](https://huggingface.co/FakeRockert543/gemma-4-31b-it-MLX-bf16) | 31B dense | bf16 | 62.5 GB | — |

## Quantization Details

- **Bits:** 8
- **Group size:** 64
- **Mode:** affine
- **Strategy:** PLE-safe — only large `nn.Linear` and `SwitchLinear` (MoE) layers are quantized. All PLE/ScaledLinear/vision/audio layers stay in bf16.

## Why PLE-Safe?

Gemma 4 uses a novel **PLE (Per-Layer Embeddings)** architecture with `ScaledLinear` layers that multiply outputs by a learned scalar. Standard quantization introduces rounding error in these layers, and the scalar amplifies it — producing `ionoxffionoxff...` garbage.

**Our fix:** Only quantize the large decoder `nn.Linear` and `SwitchLinear` (MoE expert) layers. Everything else stays bf16:

| Quantized (8bit) | Kept in bf16 |
|---|---|
| Attention projections (q/k/v/o_proj) | ScaledEmbedding (embed_tokens) |
| MLP layers (gate/up/down_proj) | ScaledLinear (PLE pathway) |
| MoE expert layers (SwitchLinear) | Per-layer embeddings (per_layer_*) |
| | Vision encoder |
| | Audio encoder || | All norms and scalars |

## Usage

**Prerequisite:** Apply the ScaledLinear fix to mlx-vlm (required until PR merged upstream):

```bash
pip install mlx-vlm

# Apply fix
git clone https://github.com/FakeRocket543/mlx-gemma4.git
cp mlx-gemma4/mlx_vlm_patches/models/gemma4/language.py \
   $(python -c "import mlx_vlm; print(mlx_vlm.__path__[0])")/models/gemma4/
```

**Important:** You must manually apply the chat template. `mlx_vlm.generate()` does not do this automatically for Gemma 4.

### Vision

```python
from mlx_vlm import load, generate

model, processor = load("FakeRockert543/gemma-4-e4b-it-MLX-8bit")
tokenizer = processor.tokenizer

messages = [{"role": "user", "content": [
    {"type": "image", "url": "photo.jpg"},
    {"type": "text", "text": "Describe this image in detail."},
]}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, ["photo.jpg"],
    max_tokens=200, repetition_penalty=1.2, temperature=0.7)
print(out.text)
```

### Audio

```python
messages = [{"role": "user", "content": [
    {"type": "audio", "url": "speech.wav"},
    {"type": "text", "text": "What is the speaker saying? What is their emotional tone?"},
]}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, audio=["speech.wav"],
    max_tokens=200, repetition_penalty=1.2, temperature=0.1)
print(out.text)
```

### Text

```python
messages = [{"role": "user", "content": "What is the capital of France?"}]
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
python convert_gemma4.py E4B 8
```

## Validation

All 12 variants validated on 10 images + 12 audio samples + 3 chat prompts. Full results: [GitHub](https://github.com/FakeRocket543/mlx-gemma4).

## License

Model weights: [Google Gemma License](https://ai.google.dev/gemma/terms). Scripts: MIT.
