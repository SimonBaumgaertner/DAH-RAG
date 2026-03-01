# benchmark_encoders.py
import time
import numpy as np

from common.strategies.encoding import QwenEncoder, MiniLMMeanPoolingEncoder

# --- prepare encoders ---
qwen_encoder = QwenEncoder(task="query", max_length=1200, model_name="Qwen/Qwen3-Embedding-0.6B")
minilm_encoder = MiniLMMeanPoolingEncoder()

# --- generate ~1000 tokens of text ---
# (rough: MiniLM tokenizes ~1.3 tokens per word on average)
base = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
)
long_text = " ".join([base] * 10)

print(f"Input text length (chars): {len(long_text)}")

# --- time Qwen ---
start = time.perf_counter()
vec_qwen = qwen_encoder.encode(long_text, query=False)
elapsed_qwen = time.perf_counter() - start

# --- time MiniLM ---
start = time.perf_counter()
vec_minilm = minilm_encoder.encode(long_text, query=False)
elapsed_minilm = time.perf_counter() - start

# --- report ---
print("\n--- Results ---")
print(f"QwenEncoder:  {vec_qwen.shape}, {elapsed_qwen:.3f}s")
print(f"MiniLMEncoder:{vec_minilm.shape}, {elapsed_minilm:.3f}s")

long_text = " ".join([base] * 100)

print(f"Input text length (chars): {len(long_text)}")

# --- time Qwen ---
start = time.perf_counter()
vec_qwen = qwen_encoder.encode(long_text, query=False)
elapsed_qwen = time.perf_counter() - start

# --- time MiniLM ---
start = time.perf_counter()
vec_minilm = minilm_encoder.encode(long_text, query=False)
elapsed_minilm = time.perf_counter() - start

# --- report ---
print("\n--- Results ---")
print(f"QwenEncoder:  {vec_qwen.shape}, {elapsed_qwen:.3f}s")
print(f"MiniLMEncoder:{vec_minilm.shape}, {elapsed_minilm:.3f}s")
