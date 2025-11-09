#!/usr/bin/env python3
"""
Script to download PubMed abstracts from HuggingFace and convert to TokAlign JSONL format.
Target: ~1 billion tokens (approximately 3.3 million abstracts)
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Try to import datasets, install if not available
try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

# Configuration
MAIN_DIR = Path(__file__).parent.parent
OUTPUT_FILE = MAIN_DIR / "data" / "pretrain-corpus" / "pubmed-corpus.json"
TARGET_TOKENS = 1_000_000_000  # 1 billion tokens
TOKENIZER_NAME = "EleutherAI/pythia-1b"  # Use Pythia tokenizer for counting


def main():
    """Main function to download and format PubMed abstracts."""
    print("=" * 70)
    print("PubMed Abstract Corpus Preparation from HuggingFace")
    print("=" * 70)
    
    # Load tokenizer for counting tokens
    print(f"\n🔤 Loading tokenizer: {TOKENIZER_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        sys.exit(1)
    
    # Load dataset from HuggingFace
    print(f"\n📥 Loading dataset from HuggingFace: uiyunkim-hub/pubmed-abstract")
    print("   Note: You may need to login using `huggingface-cli login` if the dataset is gated")
    try:
        dataset = load_dataset("uiyunkim-hub/pubmed-abstract", split="train")
        print(f"✓ Dataset loaded: {len(dataset):,} abstracts")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("   Try running: huggingface-cli login")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Process abstracts and write to JSONL format
    print(f"\n📝 Converting to JSONL format...")
    print(f"   Target: {TARGET_TOKENS:,} tokens")
    print(f"   Output: {OUTPUT_FILE}\n")
    
    total_tokens = 0
    total_abstracts = 0
    skipped_empty = 0
    skipped_short = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as jsonfile:
        for item in tqdm(dataset, desc="Processing abstracts"):
            # Extract abstract field
            abstract = item.get('abstract', '').strip()
            
            # Skip empty abstracts
            if not abstract:
                skipped_empty += 1
                continue
            
            # Skip very short abstracts (likely errors or placeholders)
            if len(abstract) < 50:  # Less than 50 characters
                skipped_short += 1
                continue
            
            # Count tokens using tokenizer
            tokens = tokenizer.encode(abstract, add_special_tokens=False)
            token_count = len(tokens)
            
            # Write JSONL format (one JSON object per line)
            # Format matches the lang-code-math-mix file: {"text": "abstract..."}
            json_obj = {"text": abstract}
            jsonfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            
            total_tokens += token_count
            total_abstracts += 1
            
            # Check if we've reached target
            if total_tokens >= TARGET_TOKENS:
                print(f"\n✓ Reached target of {TARGET_TOKENS:,} tokens!")
                break
    
    print(f"\n{'='*70}")
    print("✓ Process completed!")
    print(f"{'='*70}")
    print(f"\n📊 Statistics:")
    print(f"   Total abstracts: {total_abstracts:,}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Skipped (empty): {skipped_empty:,}")
    print(f"   Skipped (too short): {skipped_short:,}")
    print(f"   Output file: {OUTPUT_FILE}")
    print(f"   File size: {OUTPUT_FILE.stat().st_size / (1024**3):.2f} GB")
    
    if total_tokens < TARGET_TOKENS:
        print(f"\n⚠ Note: Collected {total_tokens:,} tokens (target: {TARGET_TOKENS:,})")
        print(f"   This is normal - the dataset has {len(dataset):,} total abstracts")
        print(f"   You can use all available abstracts or adjust TARGET_TOKENS")
    
    print(f"\n📁 Output file ready: {OUTPUT_FILE}")
    print(f"   You can now use this file in convert2glove_corpus.sh")


if __name__ == "__main__":
    main()

