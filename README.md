# Hate and manipulation detection using transformers

Transformer-based classifiers for hate or emotional manipulation detection

This respository contains research code for AICP-FIMI (_AI driven cloud platform to counter FIMI during elections and early warning service for identification of social media bot and troll farms_) project

**Requirements**
- Python 3.9+ (3.12 or 3.13 recommended)
- Install Python deps: `pip install -r requirements.txt` (see `requirements.txt`)

Project layout (important files)
- `metahate_clean` - code for cleaning and applying Metahate dataset
- `experiment.py` — data loading and text preprocessing utilities (cleaning, emoji handling, date removal)
- `finetune.py` — generic HuggingFace `Trainer`-based training loop, LoRA/PEFT integration, and evaluation helpers.
- `finetune_mbert.py` — training wrapper using `unsloth.FastModel` for large/long-context models and optimized PEFT. Unsloth library must be installed to run this scripts (`pip install unsloth`)
- `transformer_classifier.py` — a `TransformerClassifier` model class, cross-validation harness, tokenization and
evaluation utilities.
- `requirements.txt` — Python dependency list for this folder.

Copyright (C) 2026 Paulius Danėnas, Kaunas University of Technology




