# Pharos-CY Adapting & Serving LLMs in Production

**Cyprus University of Technology (CUT)**  
**Instructors:** Yiannos Vyrides · Kyriakos-Evmenios N. Nomikos

Participant materials for a practical workshop on adapting and serving large language models. The workshop combines an interactive lesson, a guided Google Colab notebook, presentation slides, and optional standalone GPU examples.

## Start here

1. Download the repository using **Code → Download ZIP**, then extract it.
2. Open [the interactive lesson](lessons/Session1_Student.html) locally in your browser.
3. Upload [session1.ipynb](notebooks/session1.ipynb) to [Google Colab](https://colab.research.google.com/), save a copy in Drive, and select a **T4 GPU**.
4. Run the notebook cells in order, starting with its setup instructions.

See [START_HERE.md](START_HERE.md) for setup and [the presentation](slides/Slides.pptx) for the session slides. GitHub displays HTML source rather than running the interactive lesson, so open the downloaded HTML file.

## What you will learn

- Supervised fine-tuning and common failure modes, including forgetting and style overfitting.
- LoRA and QLoRA, trainable parameter counts, and adapter size.
- GRPO group advantages, reward design, and reward hacking.
- Decoding strategies and KV-cache memory budgeting.
- PagedAttention, continuous batching, and vLLM serving.
- Retrieval-augmented generation, citations, and handling insufficient evidence.

The practical model is **Qwen/Qwen2.5-0.5B-Instruct**. Larger-model numbers in the materials are illustrative examples.

## Repository structure

```text
.
├── README.md
├── START_HERE.md
├── slides/
│   └── Slides.pptx                 # Participant presentation
├── lessons/
│   ├── Session1_Student.html        # Self-contained interactive lesson
│   └── sources/                    # Editable lesson and printable coursebook
├── notebooks/
│   └── session1.ipynb               # Main Colab / Jupyter practical
└── labs/
    ├── README.md                   # Optional standalone setup
    ├── requirements.txt            # Local / HPC training dependencies
    ├── train_qlora.py               # Standalone QLoRA example
    └── serve_vllm_rag.py            # Separate serving example
```

## Optional standalone labs

Use [the labs guide](labs/README.md) for local GPU or HPC workflows. The notebook manages its own Colab environment. Keep vLLM in a **separate serving environment**, following the dependency instructions in the materials.

For a printable lesson summary, open [the coursebook](lessons/sources/Session1_Coursebook_Source.dc.html) in your browser and print to PDF. Keep its accompanying JavaScript files in the same folder.
