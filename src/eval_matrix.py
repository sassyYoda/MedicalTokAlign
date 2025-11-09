import json
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import argparse

def read_tsv(file_path):
    res = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            res.append([d.split(" ") for d in line.strip().split("\t")])
    return res

# BLEU-1
def eval_trans_matrix(
    trans_dict_path="./log/pythia2gemma/glove-MX1M-iter15-d300.json", 
    eval_file_path="./data/pretrain-dataset/pythia-2-gemma-MX1K-eval",
    bleu_weights=(1, 0, 0, 0),
    tokenizer_path="EleutherAI/pythia-1b",
):
    with open(trans_dict_path, "r") as f:
        trans = json.load(f)

    eval_data = read_tsv(eval_file_path)

    tgt_len = len(list(trans.keys()))

    td = trans

    # Convert alignment matrix keys to strings if needed (JSON loads keys as strings)
    # Check key format
    sample_key = list(td.keys())[0] if td else None
    if sample_key and isinstance(sample_key, str):
        # Keys are strings, which is what we need
        pass
    else:
        # Convert to string keys if they're integers
        td = {str(k): v for k, v in td.items()}

    # Diagnostics
    total_tokens = 0
    missing_tokens = 0
    total_b = 0
    sample_count = 0
    
    # Analyze mapping distribution
    from collections import Counter, defaultdict
    mapping_counter = Counter()  # Count occurrences: how many times BioGPT tokens map to each Pythia token
    unique_mapped_tokens = set()
    unique_bio_to_pythia = defaultdict(set)  # Track unique mappings: which BioGPT token IDs map to which Pythia tokens
    
    # for s in tqdm(eval_data):
    for s in eval_data:
        # src: source token id, e.g., pythia ids, tgt: target token id, e.g., gemma ids
        src, tgt = s[0], s[1]
        
        # Count missing tokens
        for tid in tgt:
            total_tokens += 1
            if tid not in td:
                missing_tokens += 1

        # using td dict by maping target ids to source ids
        pred = [str(td.get(tid, "<UNK>")) for tid in tgt]  # Use .get() to handle missing tokens
        
        # Track mapping distribution (occurrences)
        for mapped_token in pred:
            if mapped_token != "<UNK>":
                mapping_counter[mapped_token] += 1
                unique_mapped_tokens.add(mapped_token)
        
        # Track unique token ID mappings
        for bio_token_id, pythia_token_id in zip(tgt, pred):
            if pythia_token_id != "<UNK>":
                unique_bio_to_pythia[pythia_token_id].add(bio_token_id)
        
        total_b += sentence_bleu([src], pred, bleu_weights)
        
        # Print first few examples for debugging
        if sample_count < 3:
            print(f"\nSample {sample_count + 1}:")
            print(f"  Source (Pythia) tokens: {src[:10]}...")  # First 10 tokens
            print(f"  Target (BioGPT) tokens: {tgt[:10]}...")
            print(f"  Predicted (mapped): {pred[:10]}...")
            sample_count += 1
    
    # Calculate mapping statistics
    total_mappings = sum(mapping_counter.values())
    most_common_mappings = mapping_counter.most_common(10)
    
    print(f"\nDiagnostics:")
    print(f"  Alignment matrix size: {len(td)} unique BioGPT token IDs")
    print(f"  Total target token occurrences in eval: {total_tokens}")
    print(f"  Missing tokens (not in alignment): {missing_tokens} ({100*missing_tokens/max(total_tokens,1):.2f}%)")
    print(f"  Unique Pythia tokens mapped to: {len(unique_mapped_tokens)}")
    print(f"  Total token occurrences mapped: {total_mappings}")
    print(f"  Average occurrences per unique Pythia token: {total_mappings/max(len(unique_mapped_tokens),1):.2f}")
    print(f"\n  Top 10 most-mapped-to Pythia tokens (by occurrence count):")
    # Decode tokens to see what they actually are
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        for token_id, count in most_common_mappings:
            unique_bio_count = len(unique_bio_to_pythia[token_id])
            try:
                token_str = tok.decode([int(token_id)])
                # Clean up token string for display
                token_str = repr(token_str) if len(token_str) > 50 else token_str
                print(f"    Token {token_id} ('{token_str}'):")
                print(f"      - {count:,} occurrences map to it ({100*count/total_mappings:.2f}% of all occurrences)")
                print(f"      - {unique_bio_count} unique BioGPT token IDs map to it")
            except (ValueError, KeyError, IndexError):
                print(f"    Token {token_id}:")
                print(f"      - {count:,} occurrences map to it ({100*count/total_mappings:.2f}% of all occurrences)")
                print(f"      - {unique_bio_count} unique BioGPT token IDs map to it [could not decode]")
    except Exception as e:
        # If tokenizer loading fails, just print IDs
        for token_id, count in most_common_mappings:
            unique_bio_count = len(unique_bio_to_pythia[token_id])
            print(f"    Token {token_id}:")
            print(f"      - {count:,} occurrences map to it ({100*count/total_mappings:.2f}% of all occurrences)")
            print(f"      - {unique_bio_count} unique BioGPT token IDs map to it")

    print(f"\nAverage bleu: {total_b/len(eval_data)}")

    return total_b/len(eval_data)

# BERT-Score
def eval_bert_score(
    trans_dict_path="./log/pythia2gemma/glove-MX1M-iter15-d300.json",
    eval_file_path="./data/pretrain-dataset/pythia-2-gemma-MX1K-eval",
    tok_path="./data/pythia-1b",
    model_path="all-mpnet-base-v2",
):
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = SentenceTransformer(model_path)

    with open(trans_dict_path, "r") as f:
        trans = json.load(f)

    eval_data = read_tsv(eval_file_path)

    tgt_len = len(list(trans.keys()))

    td = trans

    tgt_len = len(list(trans.keys()))

    total_b = 0
    
    all_src = []
    all_tgt = []
    # for s in tqdm(eval_data):
    for s in eval_data:
        src, tgt = s[0], s[1]

        # td map source token id to target token id
        # pred = [int(td[sid]) for sid in src]
        # tgt = [int(tid) for tid in tgt]

        # td map target token id to source token id
        pred = [int(td[tid]) for tid in tgt]
        tgt = [int(sid) for sid in src]

        res = tok.batch_decode([pred, tgt])
        all_src.append(res[0])
        all_tgt.append(res[1])
    
    embed0 = model.encode(all_src)
    embed1 = model.encode(all_tgt)

    sims = model.similarity(embed0, embed1)
    sim_d = []
    for i in range(len(eval_data)):
        sim_d.append(sims[i][i].item())

    return sum(sim_d)/len(sim_d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluate-method", type=str, default="bleu")
    parser.add_argument("-m", "--one2one-matrix-path", type=str, default="./data/pythia2gemma/glove.json")
    parser.add_argument("-f", "--eval-file-path", type=str, default="./data/pretrain-dataset/pythia-2-gemma-MX1K-eval")
    parser.add_argument("-t", "--tokenizer-path", type=str, default="EleutherAI/pythia-1b")
    parser.add_argument("-b", "--bert-score-model-path", type=str, default="all-mpnet-base-v2")
    parser.add_argument("-w", "--bleu-weights", type=str, default="1,0,0,0")

    args = parser.parse_args()

    if args.evaluate_method.lower() == "bleu":
        weights = tuple([float(i) for i in args.bleu_weights.split(",")])
        assert len(weights) == 4, "There are only 4 BLEU weights (BLEU-1 to 4)"
        eval_trans_matrix(
            trans_dict_path = args.one2one_matrix_path,
            eval_file_path = args.eval_file_path,
            bleu_weights = weights,
            tokenizer_path = args.tokenizer_path
        )
    elif args.evaluate_method.lower() == "bert-score" or args.evaluate_method.lower() == "bertscore":
        eval_bert_score(
            trans_dict_path = args.one2one_matrix_path,
            eval_file_path = args.eval_file_path,
            tok_path = args.tokenizer_path,
            model_path = args.bert_score_model_path,
        )
    else:
        raise Exception(f"{args.evaluate_method} is not implemented.")
