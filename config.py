import torch


def get_config():
    """
    Returns a dictionary of configuration settings.

    Returns:
        dict: Dictionary containing various configuration parameters.
    """
    return {
        "batch_size": 10,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 300,
        "d_model": 512,
        "datasource": "findnitai/english-to-hinglish",
        "lang_src": "en",
        "lang_tgt": "hi_ng",
        "tokenizer_file": "tokenizer_{0}.json",
        "num_workers": 8,
        "pin_memory": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pretrained_weights": "weights\\pretrained.pt",
    }
