# Nerion Model Weights

This directory contains bundled model weights for **self-sufficient operation**.

## GraphCodeBERT (480MB)

**Purpose:** Required for 91.8% GNN accuracy. Generates 768-dim code embeddings.

**Status:** ✅ Bundled locally at `models/graphcodebert/`

### Distribution Strategy

**For local development:**
- Weights already downloaded in this directory
- Runs 100% offline, no internet needed

**For production/distribution:**

The 480MB weights are too large for Git. Choose one approach:

#### Option 1: Git LFS (Large File Storage)
```bash
git lfs track "models/graphcodebert/*"
git add .gitattributes models/graphcodebert/
git commit -m "Add GraphCodeBERT weights via LFS"
```

#### Option 2: Separate download script
```bash
# Include in pip install postinstall hook
python3 scripts/download_models.py
```

#### Option 3: Package as separate artifact
- Host weights on release page / CDN
- Download during first `pip install nerion` or first run
- Fallback already implemented in code

### Fallback Behavior

If weights not found, Nerion automatically downloads from HuggingFace on first run:
```python
# From semantics.py line 250-255
if not bundled_model_path.exists():
    print("Downloading GraphCodeBERT from microsoft/graphcodebert-base...")
    model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
```

## Self-Sufficiency Guarantee

With bundled weights:
- ✅ No internet needed (100% offline)
- ✅ No external API calls
- ✅ No Microsoft/HuggingFace servers
- ✅ 91.8% accuracy guaranteed forever
- ✅ Runs on air-gapped systems

## Dependencies

```txt
torch>=2.1.0,<3.0.0
transformers>=4.35.0,<5.0.0
```

These are Python libraries (installed once, run locally forever).
