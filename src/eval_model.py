import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from datasets import load_from_disk
import math


def calculate_perplexity(model, tokenizer, texts, max_length=512, device="cuda"):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Accumulate
            num_tokens = inputs["input_ids"].shape[1]
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, device="cuda"):
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt from output
    generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def evaluate_on_dataset(model_path, dataset_path, num_samples=100, max_length=512, device="cuda"):

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Get text samples
    if "text" in dataset.column_names:
        texts = [item["text"] for item in dataset.select(range(min(num_samples, len(dataset))))]
    else:
        # Try to get first column
        key = dataset.column_names[0]
        texts = [item[key] for item in dataset.select(range(min(num_samples, len(dataset))))]
    
    print(f"Evaluating on {len(texts)} samples...")
    perplexity, avg_loss = calculate_perplexity(model, tokenizer, texts, max_length=max_length, device=device)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"{'='*60}\n")
    
    return perplexity, avg_loss


def generate_samples(model_path, prompts, max_length=150, temperature=0.7, top_p=0.9, device="cuda"):

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"\n{'='*60}")
    print(f"Text Generation Samples:")
    print(f"{'='*60}\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        generated = generate_text(model, tokenizer, prompt, max_length=max_length, 
                                 temperature=temperature, top_p=top_p, device=device)
        print(f"Generated: {generated}")
        print(f"{'-'*60}\n")


def main():

    parser = argparse.ArgumentParser(description="Evaluate trained vocabulary adaptation model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to evaluation dataset (for perplexity)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for perplexity evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="JSON file with prompts for generation (list of strings)")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                       help="Prompts for generation (space-separated)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Evaluate perplexity if dataset provided
    if args.dataset_path:
        evaluate_on_dataset(
            args.model_path,
            args.dataset_path,
            num_samples=args.num_samples,
            max_length=args.max_length,
            device=args.device
        )
    
    # Generate samples
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = json.load(f)
    elif args.prompts:
        prompts = args.prompts
    else:
        raise Exception("No prompts provided")
    
    generate_samples(
        args.model_path,
        prompts,
        max_length=150,
        device=args.device
    )


if __name__ == "__main__":
    main()

