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

### Datasets

The experiments utilize the following datasets:

- **Lithuanian Emotion Corpus (`corpus_emotions.json`)**:
  - **Description**: A dataset of Lithuanian comments (delfi.lt LITIS corpus) annotated for emotionality.
  - **Size**: 2,660 entries.
  - **Annotation**: Each entry contains a `content` string and an `emotionality` score ranging from 0.0 (neutral) to 2.0 (extreme).
  - **Classification Tasks**: The numerical scores are discretized into classes for different experiments:
    - **2-class**: Neutral (score 0.0) and Present (score &ge; 0.5).
    - **3-class**: Neutral (score 0.0), Medium (0.5 &ndash; 1.0), and High (score &ge; 1.5).

- **MetaHate Dataset**:
  - **Description**: A dataset focused on hate speech and manipulation detection which unifies over 30 existing datasets. It contains 1,226,202 social media posts and is formatted in TSV (Tab-Separated Values). The meta-collection features two columns: one for the hate speech label (1 for hate, 0 for non-hate) and another for the post content.
  - **Preprocessing**: The dataset is cleaned using the scripts in `clean_metahate/` to remove duplicates, non-English items and redundant entries before use in training and evaluation.

### Results

### Model Descriptions

The following transformer models were used in the experiments:

- **ModernBERT (`VSSA-SDSA/LT-MLKM-modernBERT`)**: A modern iteration of the BERT architecture incorporating recent advancements in transformer design, such as Flash Attention, Rotary Positional Embeddings (RoPE), and GeGLU activations. `LT-MLKM-modernBERT` is a Lithuanian-specific model optimized for Lithuanian language tasks and longer contexts (up to 8192 tokens).
- **LaBSE (`sentence-transformers/LaBSE`)**: A model developed by Google and optimized to produce similar embeddings for bilingual sentence pairs that are translations of each other. It supports 109 languages and is highly effective for cross-lingual tasks and multilingual sentence similarity.
- **E5 (`intfloat/multilingual-e5-large`)**: A family of state-of-the-art text embedding models trained using weakly-supervised contrastive pre-training. The multilingual variants are particularly strong in retrieval and classification tasks across diverse languages.
- **MiniLM (`sentence-transformers/all-MiniLM-L12-v2`)**: A compressed version of the BERT/RoBERTa architectures that maintains high performance while being significantly smaller and faster. `all-MiniLM-L12-v2` is widely used for efficient sentence embeddings and classification where computational resources are limited.

### Emotion Detection (Lithuanian)

Results of emotion detection (presence of emotion) in Lithuanian text using various transformer models and discretization strategies. They are obtained using stratified hooldout testing, 70% data for training, 15% data for validation and 15% data for testing.

In this experiment we considered difefernt settings, like using Lora and 4-bit quantization for more efficient finetuning (for the ModernBERT model), as well as weighted binary cross entropy to handle imbalanced dataset issue. 

#### 2-class classification (neutral vs. present)

| Model | Accuracy | Precision (Present) | Recall (Present) | F1-score (Present) | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LT-MLKM-modernBERT+Lora | 0.75 | 0.79 | 0.89 | 0.84 | 0.70 | 0.65 | 0.66 |
| LT-MLKM-modernBERT+Lora+4bit | 0.72 | 0.74 | 0.96 | 0.83 | 0.66 | 0.55 | 0.53 |
| LT-MLKM-modernBERT | 0.76 | 0.78 | 0.92 | 0.84 | 0.70 | 0.64 | 0.65 |
| LaBSE | 0.74 | 0.82 | 0.81 | 0.81 | 0.68 | 0.69 | 0.68 |
| LaBSE+WeightedBCE | 0.73 | 0.80 | 0.84 | 0.82 | 0.67 | 0.65 | 0.66 |
| e5-large | 0.71 | 0.72 | 0.98 | 0.83 | 0.61 | 0.52 | 0.46 |
| e5-large+WeightedBCE | 0.71 | 0.71 | 1.00 | 0.83 | 0.36 | 0.50 | 0.42 |
| all-MiniLM-L12 | 0.71 | 0.71 | 1.00 | 0.83 | 0.61 | 0.50 | 0.42 |
| all-MiniLM-L12+WeightedBCE | 0.70 | 0.73 | 0.92 | 0.82 | 0.59 | 0.54 | 0.52 |

#### 3-class classification (neutral, medium, high)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| VSSA-SDSA/LT-MLKM-modernBERT+Lora | 0.66 | 0.58 | 0.43 | 0.44 |
| VSSA-SDSA/LT-MLKM-modernBERT+Lora+4bit | 0.65 | 0.56 | 0.45 | 0.47 |
| VSSA-SDSA/LT-MLKM-modernBERT | 0.67 | 0.61 | 0.47 | 0.50 |
| LaBSE | 0.66 | 0.43 | 0.41 | 0.39 |
| LaBSE+WeightedBCE | 0.58 | 0.48 | 0.43 | 0.43 |
| E5 | 0.67 | 0.59 | 0.46 | 0.46 |
| E5+WeightedBCE | 0.62 | 0.51 | 0.43 | 0.44 |
| all-MiniLM-L12 | 0.63 | 0.21 | 0.33 | 0.26 |
| all-MiniLM-L12+WeightedBCE | 0.51 | 0.34 | 0.34 | 0.33 |

### MetaHate Dataset Results

Results on the MetaHate dataset using the multilingual E5 model.

| Model | Accuracy | F1-score | Precision | Recall | Kappa | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| E5-multi | 0.8248 | 0.4602 | 0.6418 | 0.3587 | 0.3655 | 0.7454 |
| E5-multi-finetune | 0.9049 | 0.7612 | 0.7978 | 0.7279 | 0.7021 | 0.8639 |

### Copyright

Copyright (C) 2026 Paulius Danėnas, Kaunas University of Technology