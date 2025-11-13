"""
Flash attention patch for LLaMA models.
This is a no-op for non-LLaMA models like GPT-NeoX (Pythia).
The actual flash attention is handled by attn_implementation="flash_attention_2" 
when loading the model.
"""


def replace_llama_attn_with_flash_attn():
    """
    Replace LLaMA attention with flash attention.
    This is a no-op for GPT-NeoX models (like Pythia) since they use
    attn_implementation="flash_attention_2" directly when loading.
    """
    # No-op: Pythia (GPT-NeoX) uses flash attention via attn_implementation parameter
    # This function exists to satisfy the import but doesn't need to do anything
    pass

