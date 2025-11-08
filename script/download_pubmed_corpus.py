#!/usr/bin/env python3
"""
Script to download PubMed abstracts using pubget and convert to TokAlign JSONL format.
Target: ~1 billion tokens (approximately 3.3 million abstracts)
"""

import os
import json
import csv
import subprocess
import sys
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Configuration
MAIN_DIR = Path(__file__).parent.parent
PUBGET_DATA_DIR = MAIN_DIR / "data" / "pubget_data"
OUTPUT_FILE = MAIN_DIR / "data" / "pretrain-corpus" / "pubmed-corpus.json"
TARGET_TOKENS = 1_000_000_000  # 1 billion tokens
TOKENIZER_NAME = "EleutherAI/pythia-1b"  # Use Pythia tokenizer for counting

# PubMed query to get all articles with abstracts
# Using a broad query to maximize coverage
PUBMED_QUERY = "*[All Fields] AND hasabstract[filter]"


def check_pubget_installed():
    """Check if pubget is installed."""
    try:
        result = subprocess.run(
            ["pubget", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ pubget is installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ pubget is not installed. Installing...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pubget"],
                check=True
            )
            print("✓ pubget installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install pubget. Please install manually: pip install pubget")
            return False


def download_pubmed_abstracts(output_dir, query, max_articles=None):
    """
    Download PubMed abstracts using pubget.
    
    Args:
        output_dir: Directory to save pubget data
        query: PubMed search query
        max_articles: Maximum number of articles to download (None for all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading PubMed abstracts...")
    print(f"   Query: {query}")
    print(f"   Output directory: {output_dir}")
    
    # Build pubget command
    cmd = [
        "pubget",
        "run",
        str(output_dir),
        "-q", query
    ]
    
    if max_articles:
        cmd.extend(["--max-articles", str(max_articles)])
    
    print(f"\n   Running: {' '.join(cmd)}\n")
    
    try:
        # Run pubget (this may take a long time)
        result = subprocess.run(cmd, check=True, text=True)
        print("✓ Download completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        return False


def find_text_csv(pubget_dir):
    """Find the text.csv file in pubget output directory."""
    pubget_dir = Path(pubget_dir)
    
    # pubget creates a structure like: query_<hash>/subset_allArticles_extractedData/text.csv
    for query_dir in pubget_dir.glob("query_*"):
        extracted_dir = query_dir / "subset_allArticles_extractedData"
        text_csv = extracted_dir / "text.csv"
        if text_csv.exists():
            return text_csv
    
    # Alternative structure
    for extracted_dir in pubget_dir.rglob("subset_allArticles_extractedData"):
        text_csv = extracted_dir / "text.csv"
        if text_csv.exists():
            return text_csv
    
    return None


def extract_abstracts_from_csv(csv_path, output_json_path, tokenizer, target_tokens):
    """
    Extract abstracts from pubget CSV and convert to JSONL format.
    Only includes abstracts (filters out full text).
    
    Args:
        csv_path: Path to pubget text.csv file
        output_json_path: Path to output JSONL file
        tokenizer: Tokenizer for counting tokens
        target_tokens: Target number of tokens to collect
    """
    csv_path = Path(csv_path)
    output_json_path = Path(output_json_path)
    
    # Create output directory if needed
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📝 Extracting abstracts from: {csv_path}")
    print(f"   Output: {output_json_path}")
    print(f"   Target: {target_tokens:,} tokens\n")
    
    total_tokens = 0
    total_abstracts = 0
    skipped_empty = 0
    skipped_short = 0
    
    # Open CSV and process
    with open(csv_path, 'r', encoding='utf-8') as csvfile, \
         open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        
        reader = csv.DictReader(csvfile)
        
        # Check if 'abstract' column exists
        if 'abstract' not in reader.fieldnames:
            print(f"✗ Error: 'abstract' column not found in CSV")
            print(f"   Available columns: {reader.fieldnames}")
            return False
        
        print("   Processing abstracts...")
        
        for row in tqdm(reader, desc="   Extracting"):
            # Extract abstract field
            abstract = row.get('abstract', '').strip()
            
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
            json_obj = {"text": abstract}
            jsonfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            
            total_tokens += token_count
            total_abstracts += 1
            
            # Check if we've reached target
            if total_tokens >= target_tokens:
                print(f"\n✓ Reached target of {target_tokens:,} tokens!")
                break
    
    print(f"\n📊 Summary:")
    print(f"   Total abstracts: {total_abstracts:,}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Skipped (empty): {skipped_empty:,}")
    print(f"   Skipped (too short): {skipped_short:,}")
    print(f"   Output file: {output_json_path}")
    print(f"   File size: {output_json_path.stat().st_size / (1024**3):.2f} GB")
    
    if total_tokens < target_tokens:
        print(f"\n⚠ Warning: Only collected {total_tokens:,} tokens (target: {target_tokens:,})")
        print(f"   You may need to download more articles or adjust the query.")
    
    return True


def main():
    """Main function to orchestrate the download and conversion process."""
    print("=" * 70)
    print("PubMed Abstract Corpus Downloader for TokAlign")
    print("=" * 70)
    
    # Check if pubget is installed
    if not check_pubget_installed():
        sys.exit(1)
    
    # Load tokenizer for counting tokens
    print(f"\n🔤 Loading tokenizer: {TOKENIZER_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        sys.exit(1)
    
    # Step 1: Download PubMed abstracts
    print(f"\n{'='*70}")
    print("STEP 1: Downloading PubMed abstracts")
    print(f"{'='*70}")
    
    # Check if pubget data already exists
    text_csv = find_text_csv(PUBGET_DATA_DIR)
    
    if text_csv and text_csv.exists():
        print(f"✓ Found existing pubget data: {text_csv}")
        response = input("   Use existing data? (y/n): ").strip().lower()
        if response != 'y':
            if not download_pubmed_abstracts(PUBGET_DATA_DIR, PUBMED_QUERY):
                sys.exit(1)
            text_csv = find_text_csv(PUBGET_DATA_DIR)
    else:
        if not download_pubmed_abstracts(PUBGET_DATA_DIR, PUBMED_QUERY):
            sys.exit(1)
        text_csv = find_text_csv(PUBGET_DATA_DIR)
    
    if not text_csv or not text_csv.exists():
        print(f"✗ Error: Could not find text.csv in {PUBGET_DATA_DIR}")
        print("   Please check the pubget output directory structure.")
        sys.exit(1)
    
    # Step 2: Extract abstracts and convert to JSONL
    print(f"\n{'='*70}")
    print("STEP 2: Extracting abstracts and converting to JSONL")
    print(f"{'='*70}")
    
    if not extract_abstracts_from_csv(
        text_csv,
        OUTPUT_FILE,
        tokenizer,
        TARGET_TOKENS
    ):
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("✓ Process completed successfully!")
    print(f"{'='*70}")
    print(f"\n📁 Output file: {OUTPUT_FILE}")
    print(f"   You can now use this file in convert2glove_corpus.sh")
    print(f"   Update TRAIN_FILE to: {OUTPUT_FILE.relative_to(MAIN_DIR)}")


if __name__ == "__main__":
    main()

