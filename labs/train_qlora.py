"""QLoRA fine-tune — standalone version of session1.ipynb Part A.

Runs on a single GPU (a 16 GB T4 is enough for the teaching model below).
Swap BASE for a 13B model on HPC — the 4-bit config already fits it on ~16 GB.

    pip install -r requirements.txt
    python train_qlora.py

API note: this file tracks the same TRL/Transformers surface as the notebook —
SFTConfig.max_length (renamed from max_seq_length), warmup_steps (warmup_ratio was
removed in Transformers 5) and completion_only_loss (prompt/completion loss masking).
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

BASE = "Qwen/Qwen2.5-0.5B-Instruct"     # production course path uses a 13B base
OUTPUT_DIR = "./adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Single source of truth for every dtype below: bf16 needs compute capability >= 8.0
# (Ampere+). A T4 is sm_75 and must train in fp16.
use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16   # 4-bit compute + autocast
# CPU has no autocast here and no fp16 matmul kernels, so the CPU fallback loads fp32.
model_dtype = compute_dtype if device == "cuda" else torch.float32


def build_dataset(n=800):
    """Dolly rows -> TRL's conversational prompt/completion format.

    SFTTrainer serialises each example with the tokenizer's own chat template (the same
    template used at inference) and, with completion_only_loss=True, computes the loss on
    the assistant turn only (prompt tokens are labelled -100).
    """
    ds = load_dataset("databricks/databricks-dolly-15k", split="train").select(range(n))

    def to_prompt_completion(ex):
        ctx = ex.get("context", "")
        prompt = ex["instruction"] if not ctx else f"{ex['instruction']}\n\nContext:\n{ctx}"
        return {
            "prompt":     [{"role": "user",      "content": prompt}],
            "completion": [{"role": "assistant", "content": ex["response"]}],
        }

    return ds.map(to_prompt_completion, remove_columns=ds.column_names)


def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb if device == "cuda" else None,   # 4-bit needs CUDA
        dtype=model_dtype,      # explicit: Transformers 5 defaults to dtype="auto", which
                                # reads Qwen's config (bfloat16) and would load bf16 weights
                                # even on a T4 that cannot train them.
        device_map="auto" if device == "cuda" else None,   # "auto" places/shards across GPUs;
                                # on CPU it may disk-offload layers to the meta device.
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA on the attention + MLP projections; lora_alpha / r = 2 scales the update B @ A.
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # 1 epoch, LR 2e-5, cosine scheduler + 10 warmup steps.
    trainer = SFTTrainer(
        model=model,
        peft_config=lora,
        train_dataset=build_dataset(),
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            bf16=use_bf16,
            fp16=(device == "cuda" and not use_bf16),
            logging_steps=10,
            completion_only_loss=True,   # loss on the assistant turn only
            max_length=1024,
            report_to="none",
        ),
    )

    # SFTTrainer.__init__ force-casts every trainable parameter of a 4-bit/8-bit model
    # to bfloat16 (QLoRA paper recommendation), without checking what the GPU supports.
    # On a T4 we train fp16 + GradScaler, whose unscale kernel has no bfloat16 kernel:
    #   NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
    #                        not implemented for 'BFloat16'
    # Put the adapter back in fp32 — the normal master-weight dtype for fp16 AMP.
    if not use_bf16:
        for p in trainer.model.parameters():
            if p.requires_grad and p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float32)
    trainable_dtypes = sorted({str(p.dtype) for p in trainer.model.parameters() if p.requires_grad})
    print("trainable parameter dtypes:", trainable_dtypes)
    assert use_bf16 or "torch.bfloat16" not in trainable_dtypes, (
        "bfloat16 trainable parameters on a GPU whose GradScaler cannot unscale them"
    )

    trainer.train()

    # print_trainable_parameters() lives on the PEFT model, which SFTTrainer wraps.
    trainer.model.print_trainable_parameters()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print("Saved LoRA adapter to", OUTPUT_DIR, sorted(os.listdir(OUTPUT_DIR)))


if __name__ == "__main__":
    main()
