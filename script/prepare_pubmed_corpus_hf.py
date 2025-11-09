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
from multiprocessing import Pool, cpu_count

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
BATCH_SIZE = 1000  # Process this many abstracts at once
NUM_WORKERS = None  # None = use all available CPU cores


# Global tokenizer for worker processes (initialized once per worker)
_worker_tokenizer = None

def init_worker(tokenizer_name):
    """Initialize tokenizer in each worker process."""
    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def process_abstract_batch(batch):
    """
    Process a batch of abstracts in parallel.
    
    Args:
        batch: List of dataset items
    
    Returns:
        List of (json_line, token_count) tuples
    """
    global _worker_tokenizer
    results = []
    skipped_empty = 0
    skipped_short = 0
    
    for item in batch:
        abstract = item.get('abstract', '').strip()
        
        # Skip empty abstracts
        if not abstract:
            skipped_empty += 1
            continue
        
        # Skip very short abstracts
        if len(abstract) < 50:
            skipped_short += 1
            continue
        
        # Tokenize
        tokens = _worker_tokenizer.encode(abstract, add_special_tokens=False)
        token_count = len(tokens)
        
        # Format as JSONL
        json_obj = {"text": abstract}
        json_line = json.dumps(json_obj, ensure_ascii=False)
        
        results.append((json_line, token_count))
    
    return results, skipped_empty, skipped_short


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
    print(f"   Output: {OUTPUT_FILE}")
    
    # Determine number of workers
    num_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()
    print(f"   Using {num_workers} CPU cores for parallel processing\n")
    
    total_tokens = 0
    total_abstracts = 0
    skipped_empty = 0
    skipped_short = 0
    
    # Create batches
    dataset_list = list(dataset)  # Convert to list for batching
    batches = [dataset_list[i:i + BATCH_SIZE] for i in range(0, len(dataset_list), BATCH_SIZE)]
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as jsonfile:
        # Process batches in parallel
        # Initialize each worker with the tokenizer
        with Pool(processes=num_workers, initializer=init_worker, initargs=(TOKENIZER_NAME,)) as pool:
            # Use imap for progress tracking
            results = pool.imap(process_abstract_batch, batches)
            
            for batch_results, batch_empty, batch_short in tqdm(
                results, 
                total=len(batches),
                desc="Processing batches"
            ):
                skipped_empty += batch_empty
                skipped_short += batch_short
                
                # Write results from this batch
                for json_line, token_count in batch_results:
                    jsonfile.write(json_line + "\n")
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

