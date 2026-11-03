# Optional standalone labs

The workshop's main practical is [the Colab notebook](../notebooks/session1.ipynb). These scripts provide optional local GPU or HPC examples.

| File | Purpose |
|---|---|
| `train_qlora.py` | Fine-tune a small model with QLoRA and save its adapter |
| `serve_vllm_rag.py` | Serve the adapter with vLLM and demonstrate retrieval-augmented generation |
| `requirements.txt` | Training dependencies and separate serving setup instructions |

## Training

Use a fresh Python environment and a compatible NVIDIA GPU. From this directory, install PyTorch for your CUDA environment, then install the training dependencies:

```bash
python -m venv .venv-training
source .venv-training/bin/activate
python -m pip install -r requirements.txt
python train_qlora.py
```

The script writes an `adapter/` directory. Keep it for the serving example.

**In Colab, use the notebook's setup cell instead of this requirements file.** The notebook selects FP16 on a T4 and BF16 on supported newer GPUs.

## Serving

Use a separate environment or container for vLLM. Follow the serving section in `requirements.txt`; it is intentionally separate from the training dependencies.

Copy the trained `adapter/` directory beside `serve_vllm_rag.py`, then run:

```bash
python serve_vllm_rag.py
```

The serving environment must support vLLM and its PyTorch/CUDA requirements. Keep it separate from the Colab or standalone training environment.

## Learning resources

- [Interactive student lesson](../lessons/Session1_Student.html)
- [Participant slides](../slides/Slides.pptx)
- [Workshop overview](../README.md)
