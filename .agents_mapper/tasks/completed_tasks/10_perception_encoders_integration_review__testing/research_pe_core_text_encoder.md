# Research: PE-Core CLIP Text Encoder Origin

## Question
When using PE-Core models (vision+text), which version of CLIP is being used for the text encoder?

## Summary
The text encoder is **Meta's custom implementation** based on **OpenAI CLIP's architecture and tokenizer**, but with custom transformer configurations optimized for the Perception Engine models.

---

## Findings

### 1. CLIP Class Definition

| Item | Location |
|------|----------|
| **File** | `thirdParty/perception_models/core/vision_encoder/pe.py` |
| **Class** | `CLIP` (lines 700-761) |
| **Parent** | Inherits from `TextTransformer` |

### 2. Text Encoder Implementation

The text encoder (`TextTransformer`) is defined in `pe.py:552-695`:

- **Architecture**: Transformer-based with:
  - Causal attention mask (autoregressive text encoding)
  - Layer normalization
  - MLP components
  - Optional attention pooling
  - Text projection to output dimension
- **Vocab size**: 49408 (matches OpenAI CLIP)
- **Not imported** from `openai_clip` or `open_clip` - it's a fresh implementation

### 3. Tokenizer

| Item | Details |
|------|---------|
| **File** | `thirdParty/perception_models/core/vision_encoder/tokenizer.py` |
| **Class** | `SimpleTokenizer` (lines 132-277) |
| **Origin** | OpenAI CLIP (MIT License, copyright notice at lines 1-3) |
| **Vocab file** | `bpe_simple_vocab_16e6.txt.gz` (1.3 MB) - standard OpenAI BPE vocabulary |
| **Context length** | Default 77 tokens (OpenAI standard) |
| **Special tokens** | `<start_of_text>`, `<end_of_text>` |

### 4. Model Configuration

PE-Core models have custom text encoder configs in `config.py`:

```python
# Example: PE-Core-L14-336
PE_TEXT_CONFIG["PE-Core-L14-336"] = PETextConfig(
    context_length=32,    # Different from OpenAI's 77
    width=1024,
    heads=16,
    layers=24,
    output_dim=1024
)
```

### 5. Key Differences from OpenAI CLIP

| Aspect | OpenAI CLIP | PE-Core |
|--------|-------------|---------|
| Context length | 77 tokens | 32 or 72 (model-dependent) |
| Transformer | Standard | Custom with optional RoPE |
| Weights source | OpenAI | HuggingFace (`facebook/PE-Core-*`) |
| Tokenizer | BPE | Same BPE (copied) |

---

## Conclusion

The PE-Core text encoder is:
1. **Architecturally based on OpenAI CLIP** - same tokenizer, similar transformer structure
2. **Custom implementation** - not a direct import, reimplemented in PE codebase
3. **Different weights** - trained by Meta, loaded from HuggingFace
4. **Enhanced features** - RoPE support, custom context lengths

When using PE-Core for text queries, you're using Meta's CLIP-like text encoder with Meta's trained weights, not OpenAI's original model.

---

## Files Investigated
- `thirdParty/perception_models/core/vision_encoder/pe.py`
- `thirdParty/perception_models/core/vision_encoder/tokenizer.py`
- `thirdParty/perception_models/core/vision_encoder/config.py`
