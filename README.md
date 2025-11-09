# MedicalTokAlign: Vocabulary Adaptation for Medical Domain

This repository adapts [TokAlign](https://github.com/chongli17/TokAlign) for the medical domain by adapting Pythia-1b to use BioGPT's tokenizer, trained on PubMed abstracts.

**Based on:** TokAlign: Efficient Vocabulary Adaptation via Token Alignment (ACL 2025)

## Overview
This project applies the TokAlign method to adapt Pythia-1b's vocabulary to BioGPT's tokenizer for medical domain applications. The method aligns the source vocabulary (Pythia-1b) to the target vocabulary (BioGPT) by learning a one-to-one mapping matrix for token IDs. Model parameters, including embeddings, are rearranged and progressively fine-tuned for the new vocabulary using PubMed abstracts.

**Key Differences from Original TokAlign:**
- **Base Model/Tokenizer**: Pythia-1b (maintained as in original)
- **Target Model/Tokenizer**: BioGPT (microsoft/biogpt)
- **Training Corpus**: PubMed abstracts (~1 billion tokens)

![](figure/method.png)

## How to run?

### Set up an virtual environment
```
conda create -n tokalign python=3.10
conda activate tokalign
pip install -r requirements.txt
```

### Prepare PubMed Corpus

1. **Download and format PubMed abstracts** from HuggingFace:
```bash
# The script will download from HuggingFace and format to JSONL
python script/prepare_pubmed_corpus_hf.py
```

This script will:
- Load the [PubMed Abstracts dataset](https://huggingface.co/datasets/uiyunkim-hub/pubmed-abstract) from HuggingFace
- Extract abstracts and format as JSONL: `{"text": "abstract..."}`
- Count tokens to track progress toward 1B tokens
- Save to `data/pretrain-corpus/pubmed-corpus.json`

**Note:** You may need to login to HuggingFace first:
```bash
huggingface-cli login
```

2. **Tokenize corpus and prepare files of GloVe vector training and evaluation**:
```bash
# Update paths in the script (already configured for BioGPT)
vim script/convert2glove_corpus.sh 
bash script/convert2glove_corpus.sh 
```

### Train GloVe vectors and obtain token alignment matrix

```
git clone https://github.com/stanfordnlp/GloVe.git
# Train GloVe vectors for source vocabulary and target vocabulary
bash script/token_align.sh
```

### Evaluation of one-to-one token alignment matrix learned
```
# Change the path to the alignment matrix path for evaluation, and choose an evaluation method (BLEU-1 or Bert-score).
vim script/eval_align.sh
bash script/eval_align.sh
```

### Initialize the model weight with the token alignment matrix

```
# Modify the path of alignment matrix
vim script/init_model.sh 
bash script/init_model.sh 
```

### Vocabulary Adaptation
```
# First tokenize the training dataset used for vocabulary adaptation
vim script/tokenize_dataset.sh
bash script/tokenize_dataset.sh

# Replace some paths and hyper-parameters with yours, and start the vocabulary adaptation process
vim script/vocab_adaptation.sh
bash script/vocab_adaptation.sh
```

## Configuration

This repository is configured for:
- **Base Model**: `EleutherAI/pythia-1b`
- **Target Model/Tokenizer**: `microsoft/biogpt`
- **Training Corpus**: PubMed abstracts from [HuggingFace](https://huggingface.co/datasets/uiyunkim-hub/pubmed-abstract) (~27.7M abstracts, ~1 billion tokens)

All scripts have been updated to use BioGPT instead of Gemma/Qwen2/LLaMA3. See individual scripts in `script/` directory for details.

## Scripts Overview

- `script/prepare_pubmed_corpus_hf.py` - Download and format PubMed abstracts from HuggingFace
- `script/convert2glove_corpus.sh` - Tokenize corpus for GloVe training
- `script/token_align.sh` - Train GloVe vectors and compute token alignment
- `script/eval_align.sh` - Evaluate token alignment matrix
- `script/init_model.sh` - Initialize model with alignment matrix
- `script/tokenize_dataset.sh` - Tokenize dataset for vocabulary adaptation
- `script/vocab_adaptation.sh` - Run vocabulary adaptation training

## Citation

If you use this code, please cite the original TokAlign paper:

```bibtex
@inproceedings{li-etal-2025-TokAlign,
  author    = {Chong Li and
               Jiajun Zhang and
               Chengqing Zong},
  title = "TokAlign: Efficient Vocabulary Adaptation via Token Alignment",
  booktitle = "Proceedings of the 63nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
}
```

## Acknowledgments

This repository is based on [TokAlign](https://github.com/chongli17/TokAlign) by Chong Li, Jiajun Zhang, and Chengqing Zong, adapted for medical domain applications using BioGPT and PubMed data.
