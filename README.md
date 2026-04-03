# Gemma 4 MLX — PLE-Safe Quantization

> Tested on Apple M4 Max 128GB, 2026-04-03
> Framework: mlx-vlm (+ 4 bug fixes)
> Source: https://github.com/anthropic-felix/mlx-gemma4

Working MLX quantized weights for the full Google Gemma 4 family on Apple Silicon.

**All existing MLX quantized Gemma 4 models on HuggingFace (mlx-community, unsloth) are broken.** This repo provides the first working quantized versions with full trimodal (vision + audio + text) validation.

## Why Existing Quantizations Are Broken

| Source | Issue | Symptom |
|---|---|---|
| mlx-community 4bit/8bit | PLE layers quantized | `ionoxffionoxff...` garbage output |
| unsloth 4bit | 963 multimodal weights stripped | Model fails to load |

**Root cause:** Gemma 4 uses a novel **PLE (Per-Layer Embeddings)** architecture with `ScaledLinear` layers that multiply outputs by a scalar. Standard quantization introduces error in these layers, and the scalar multiplication amplifies it catastrophically.

## PLE-Safe Quantization Strategy

Only quantize the large `nn.Linear` and `SwitchLinear` (MoE expert) layers in the decoder. Everything else stays in bf16:

| Quantized (4-bit or 8-bit) | Kept in bf16 |
|---|---|
| Attention projections (q/k/v/o_proj) | ScaledEmbedding (embed_tokens) |
| MLP layers (gate/up/down_proj) | ScaledLinear (PLE pathway) |
| MoE expert layers (SwitchLinear) | Per-layer embeddings (per_layer_*) |
| | Vision encoder |
| | Audio encoder (E2B/E4B) |
| | All norms and scalars |

### Layer-Level Precision Detail (E4B example)

| Layer Category | Count | Params (M) | Q4 | Q8 |
|---|---|---|---|---|
| language/MLP (gate/up/down_proj) | 126 | 3303.0 | 4-bit | 8-bit |
| language/attention (q/k/v/o_proj) | 168 | 642.3 | 4-bit | 8-bit |
| language/ScaledEmbedding (embed_tokens) | 2 | 3489.7 | bf16 | bf16 |
| audio_tower | 270 | 317.4 | bf16 | bf16 |
| vision_tower | 209 | 151.6 | bf16 | bf16 |
| language/PLE (per_layer_*) | 127 | 55.2 | bf16 | bf16 |
| language/ScaledLinear (PLE) | 1 | 27.5 | bf16 | bf16 |
| language/norms | 253 | 0.5 | bf16 | bf16 |
| **Quantized portion** | **294** | **3945.3 (49.4%)** | | |
| **Kept bf16** | **864** | **4047.7 (50.6%)** | | |

## Available Models

| Model | 4-bit | 8-bit | bf16 | Audio |
|---|---|---|---|---|
| **E2B** (2.3B params) | 7.6 GB | 8.5 GB | 10.2 GB | ✅ |
| **E4B** (4.5B params) | 10.3 GB | 12.3 GB | 16.0 GB | ✅ |
| **26B-A4B** (26B MoE) | 16.4 GB | 28.6 GB | 51.6 GB | — |
| **31B** (31B dense) | 20.4 GB | 35.1 GB | 62.5 GB | — |

## Validation Results

All 12 variants tested on: 10 images (caption), 12 audio questions (4 categories), 3 languages (chat).

| Model | Precision | Vision (10) | Audio (12) | Chat (3) | Pass |
|---|---|---|---|---|---|
| E2B | 4-bit | 10/10 | 12/12 | 3/3 | ✅ |
| E2B | 8-bit | 10/10 | 12/12 | 3/3 | ✅ |
| E2B | bf16 | 10/10 | 12/12 | 3/3 | ✅ |
| E4B | 4-bit | 10/10 | 12/12 | 3/3 | ✅ |
| E4B | 8-bit | 10/10 | 12/12 | 3/3 | ✅ |
| E4B | bf16 | 10/10 | 12/12 | 3/3 | ✅ |
| 26B-A4B | 4-bit | 10/10 | N/A | 3/3 | ✅ |
| 26B-A4B | 8-bit | 10/10 | N/A | 3/3 | ✅ |
| 26B-A4B | bf16 | 10/10 | N/A | 3/3 | ✅ |
| 31B | 4-bit | 10/10 | N/A | 3/3 | ✅ |
| 31B | 8-bit | 10/10 | N/A | 3/3 | ✅ |
| 31B | bf16 | 10/10 | N/A | 3/3 | ✅ |

## Audio Validation

All audio samples are real human recordings — no synthetic/TTS audio.

| File | Source | Content |
|---|---|---|
| obama_30s.wav | [HF hf-internal-testing](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples) | Obama farewell address (English) |
| en_jobs.wav | YouTube | Steve Jobs Stanford commencement (English) |
| zh_ted.wav | YouTube | Chinese TED talk (Mandarin) |
| ja_speech.wav | YouTube | Japanese speech (Japanese) |
| rickroll_30s.wav | YouTube | Rick Astley - Never Gonna Give You Up (music) |

### Transcription

| Audio | bf16 | 4-bit | 8-bit |
|---|---|---|---|
| Obama (EN) | "This week I traveled to Chicago to deliver my final farewell address..." | ✅ identical | ✅ identical |
| Jobs (EN) | "drop in for another 18 months or so before I really quit..." | ✅ identical | ✅ identical |
| Chinese TED | "One of my earliest memories is trying to wake up one of my relatives..." | ✅ identical | ✅ identical |
| Japanese | "あ、話したいな。でも話できない..." | ✅ identical | ✅ identical |

### Scene & Context Understanding

| Audio | Question | bf16 | 4-bit | 8-bit |
|---|---|---|---|---|
| Obama | Casual or formal? | "formal speech" | "formal speech" | "formal speech" |
| Rickroll | Speech or music? | "music" | "music" | "music" |

### Emotion & Tone Analysis

| Audio | bf16 | 4-bit | 8-bit |
|---|---|---|---|
| Obama | "serious and grateful" | "reflective and sincere" | "serious and grateful" |
| Jobs | "reflective, narrative, matter-of-fact" | "reflective, narrative" | "reflective, narrative, matter-of-fact" |
| Japanese | "nervous, hesitation" | "nervous, hesitation" | "nervous, hesitation" |

### Speaker Analysis

| Audio | bf16 | 4-bit | 8-bit |
|---|---|---|---|
| Chinese TED | "female, 1 speaker" | "female, 1 speaker" | "female, 1 speaker" |
| Obama | "older adult" | "male, older adult" | "older adult" |

> **Note on music:** Gemma 4's audio encoder is trained on speech only. Song/artist identification scores 0/10 on all variants including bf16. This is a model limitation, not a quantization issue.

## Vision Caption Samples

### E4B 4-bit (59 tok/s)

> **Cat on shelf:** "A cozy, high-angle view capturing a domestic scene where a fluffy cat rests on an elevated shelf next to a pet food dispenser, set against the backdrop of modern interior design..."

> **Tabby cat:** "A regal, silver-grey and white tabby cat with striking green eyes gazes upwards while perched elegantly on a wooden surface, sporting a brown collar adorned with a small mint-colored bell..."

> **Salad:** "A vibrant and fresh salad, brimming with crisp green lettuce leaves mixed with finely chopped ingredients like dark leafy greens, chunks of chicken or tuna..."

### 31B 4-bit (best quality)

> **Cat on shelf:** "A high-angle, wide shot captured by a Tapo security camera shows a white and brown tabby cat perched on a light-colored wooden shelf in the upper right corner of a room..."

> **Tabby cat:** "A high-angle, medium shot captures a curious tabby and white cat sitting upright on a brown surface against a plain white background. The cat has short, dense fur with grey and black tabby stripes..."

## Performance (Apple M4 Max 128GB)

Measured on E4B with 30 images:

| Variant | Caption tok/s | Chat tok/s | Peak Memory |
|---|---|---|---|
| 4-bit | 59 | ~90 | 11.1 GB |
| 8-bit | 47 | ~63 | 13.1 GB |
| bf16 | 44 | ~44 | 16.8 GB |

## Bugs Found in mlx-vlm

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | `ScaledLinear` inherits `nn.Module` instead of `nn.Linear` | `nn.quantize()` cannot discover or quantize these layers | Change to `ScaledLinear(nn.Linear)` |
| 2 | Standard quantization quantizes PLE layers | 4-bit/8-bit output is garbage | PLE-safe `class_predicate` that skips PLE/vision/audio |
| 3 | `processor.save_pretrained()` strips `feature_extractor` from config | Audio input silently dropped (no error, model just says "no audio provided") | Copy `processor_config.json` from source model |
| 4 | `SwitchLinear` (MoE experts) not included in quantization | 26B-A4B: 49 GB instead of 16 GB | Check `hasattr(module, 'to_quantized')` in addition to `isinstance(nn.Linear)` |

## Usage

**Prerequisite:** Apply the ScaledLinear fix to mlx-vlm (required until PR is merged upstream):

```bash
pip install git+https://github.com/Blaizzy/mlx-vlm.git@main

# Apply fix
git clone https://github.com/FakeRocket543/mlx-gemma4.git
cp mlx_gemma4/mlx_vlm_patches/models/gemma4/language.py \
   $(python -c "import mlx_vlm; print(mlx_vlm.__path__[0])")/models/gemma4/
```

**Important:** You must manually apply the chat template. `mlx_vlm.generate()` does not do this automatically for Gemma 4.

### Vision

```python
from mlx_vlm import load, generate

model, processor = load("FakeRockert543/gemma-4-E4B-it-MLX-4bit")
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
    {"type": "text", "text": "What is the speaker's emotional tone? What is the topic?"},
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

## Convert From Source

```bash
git clone https://github.com/FakeRocket543/mlx-gemma4.git
cd mlx_gemma4

# Single model
python convert_gemma4.py E4B 4     # E4B 4-bit
python convert_gemma4.py E2B 8     # E2B 8-bit
python convert_gemma4.py 31B bf16  # 31B bf16

# All 12 variants
python convert_gemma4.py all

# Validate
python validate_all.py
```

## License

Model weights are subject to [Google's Gemma license](https://ai.google.dev/gemma/terms). Quantization scripts and fixes are MIT licensed.
