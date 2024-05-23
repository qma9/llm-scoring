from dotenv import load_dotenv
from os import getenv

load_dotenv()


CHARACTER_MAX = 4880

# LLama model config
MODEL_SIZE = "13b"
MODEL = f"llama-2-{MODEL_SIZE}-chat.Q4_K_M.gguf"
MODEL_PATH = f"{getenv('MODEL_PATH')}{MODEL}"
N_CTX = 4096  # llama-2 supports up 4096 token length
N_GPU_LAYERS = 25  # change this value based on your model and your GPU VRAM pool
N_BATCH = 3192  # should be between 1 and n_ctx, depends on VRAM in your GPU

DESCRIPTOR = "company review"  # earnings call transcript

# Cultural/Structural attributes of interest
POSITIVE_ATTRIBUTES = [
    "centralized",
    "secretive",
    "hierarchical",
    "formal",
    "stagnating",
    "risk_averse",
]
NEGATIVE_ATTRIBUTES = [
    "decentralized",
    "transparent",
    "non_hierarchical",
    "informal",
    "innovative",
    "risk_taking",
]
