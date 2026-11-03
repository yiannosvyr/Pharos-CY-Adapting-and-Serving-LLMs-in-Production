"""vLLM serving + a small RAG loop — the SEPARATE serving lab for Session 1.

    ┌────────────────────────────────────────────────────────────────────┐
    │  This script does NOT run in the Colab training runtime.           │
    │  vLLM ships its own torch/CUDA build; installing it next to        │
    │  Colab's preinstalled PyTorch replaces those wheels and breaks the  │
    │  training stack ("PyTorch compiled for CUDA X / TorchAudio         │
    │  compiled for CUDA Y").  Use a dedicated machine, container or      │
    │  fresh venv for this file.                                          │
    └────────────────────────────────────────────────────────────────────┘

Environment (its own venv/container, one GPU):

    python -m venv .venv-serve && source .venv-serve/bin/activate
    pip install "vllm>=0.10" sentence-transformers
    # vllm pulls the torch build it was compiled against — let it.

Inputs:
    ./adapter   the LoRA adapter produced by session1.ipynb / train_qlora.py.
                Copy it across from the training machine (it is only a few MB).

Run:
    python serve_vllm_rag.py

dtype/gpu_memory_utilization below are sized for a 16 GB T4-class card. On an
Ampere-or-newer GPU use dtype="bfloat16" and you can raise the utilisation.
"""
import os
import re

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from sentence_transformers import SentenceTransformer, util

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "./adapter"

CORPUS = [
    "LoRA freezes W and learns a low-rank BA update; only 2*d*r parameters train.",
    "PagedAttention stores the KV-cache in non-contiguous blocks to cut waste.",
    "GRPO standardises rewards within a group so different reward scales contribute comparably.",
    "Dividing by the group standard deviation puts every prompt group on the same scale, so a "
    "high-variance group cannot dominate the update and a low-variance group still teaches the model.",
    "Nucleus sampling draws from the smallest set of tokens whose probability mass exceeds p.",
    "Nicosia is the capital of Cyprus.",
]

# Retrieval gate (same value as session1.ipynb): passages that contain the answer score
# cos_sim ~0.45-0.85 on this corpus, unrelated ones ~0.0-0.2. Below the gate a passage is
# not evidence and must not enter the prompt with a citation tag.
MIN_SIM = 0.35


def build_retriever():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    C = embed.encode(CORPUS, convert_to_tensor=True)

    def retrieve(query, k=2, min_sim=MIN_SIM):
        q = embed.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q, C)[0]
        top = scores.topk(min(k, len(CORPUS)))
        return [CORPUS[i] for s, i in zip(top.values, top.indices) if float(s) >= min_sim]

    return retrieve


def build_prompt(query, docs):
    """Same prompt as session1.ipynb: rules AFTER the context, plus two examples.

    Order matters for a small model — rules-first makes it ignore the citation rule,
    and rules-without-examples make it answer "unknown" to everything.
    """
    context = "\n".join(f"[{i}] {d}" for i, d in enumerate(docs)) or "(no relevant passages found)"
    return (
        f"Context:\n{context}\n\nQuestion: {query}\n\n"
        "Rules:\n"
        "1. Answer in one or two sentences using ONLY the context above.\n"
        "2. Finish with the citation of the item you used, e.g. [0].\n"
        "3. If the context does not contain the answer, reply with exactly: unknown\n\n"
        'Example of a grounded answer: "Beam search keeps k candidates. [1]"\n'
        'Example when the context does not help: "unknown"\n\nAnswer:'
    )


def main():
    has_adapter = os.path.isdir(ADAPTER_DIR)
    if not has_adapter:
        print(f"No adapter at {ADAPTER_DIR} — serving the base model only. "
              "Copy ./adapter across from the training run to serve the fine-tune.")

    llm = LLM(
        model=BASE,
        dtype="float16",                 # "bfloat16" on Ampere or newer
        gpu_memory_utilization=0.80,
        enable_prefix_caching=True,      # reuse the KV-cache across shared prefixes
        enable_lora=has_adapter,         # hot-swap adapters per request
        max_model_len=2048,
        # tensor_parallel_size=2,        # uncomment for 2-GPU tensor parallelism
    )

    retrieve = build_retriever()
    # Greedy (temperature 0) and no repetition penalty: a grounded answer has to copy prompt tokens.
    sp = SamplingParams(temperature=0.0, max_tokens=128, repetition_penalty=1.0)
    lora_request = LoRARequest("session1-adapter", 1, ADAPTER_DIR) if has_adapter else None

    for query in ["Why does GRPO divide by the group standard deviation?",   # answered by the course passages
                  "What is the capital of Cyprus?",                            # answered by one passage
                  "Who painted the Mona Lisa?"]:                               # not answerable -> unknown
        docs = retrieve(query)
        prompt = build_prompt(query, docs)
        raw = llm.generate([prompt], sp, lora_request=lora_request)[0].outputs[0].text.strip()
        # Citation check (same as the notebook): only an answer that cites a passage really present
        # in the context is shown; "unknown" or a citation of a non-existent passage -> "unknown".
        cited = [int(i) for i in re.findall(r"\[(\d+)\]", raw)]
        grounded = bool(cited) and all(i < len(docs) for i in cited) and not raw.lower().startswith("unknown")
        print("Q:", query)
        print("   context:", docs or "(no relevant passages found)")
        print("   raw model output:", repr(raw))
        print("   displayed answer:", raw if grounded else "unknown")


if __name__ == "__main__":
    main()
