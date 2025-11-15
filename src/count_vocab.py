from transformers import AutoConfig
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, default="google/gemma-2b")
    args = parser.parse_args()
    # Load config only (avoids loading model and flash-attn dependency)
    config = AutoConfig.from_pretrained(args.model_path)
    print(config.vocab_size)
