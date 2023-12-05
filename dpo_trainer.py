import os
import torch

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import find_all_linear_names, print_trainable_parameters

output_dir="./palmer-003"
model_name = "appvoid/palmer-002"

dataset = load_dataset("json", data_files="dpo_orca.json",split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

model_ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def return_prompt_and_responses(samples):
    return {
        "prompt": [
            f"{input}"
            for input in samples["input"]
        ],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }

original_columns = dataset.column_names

dataset = dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing =False,
    max_grad_norm= 0.3,
    num_train_epochs=15,
    save_steps= 0,
    learning_rate=1e-5,
    bf16=False,
    save_total_limit=0,
    logging_steps=100,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    remove_unused_columns=False
)

peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=1024,
    max_length=4096,
)


dpo_trainer.train()
dpo_trainer.save_model(output_dir)


output_dir = os.path.join(output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
