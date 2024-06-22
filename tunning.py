from datasets import load_dataset
from huggingface_hub import login
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

#login ..
login(token="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")

#load the modified data...
dataset = load_dataset('ar111/modified_orca_dataset')

#load the models...
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

from peft import LoraConfig,get_peft_model

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

from transformers import TrainingArguments
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    optim = "paged_adamw_32bit",
    save_steps = 100,
    logging_steps = 10,
    learning_rate = 2e-4,
    max_grad_norm = 0.3,
    # fp16=True,
    max_steps = 100,
    warmup_ratio = 0.03,
    lr_scheduler_type = "constant"
)

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)


for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  
model_to_save.save_pretrained("outputs")

lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)

model = peft_config.from_pretrained(model, 'Llama-2-7b-chat-hf-orcaTuned')
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

import locale
locale.getpreferredencoding = lambda: "UTF-8"
model.push_to_hub("ar111/Llama-2-7b-chat-hf-orcaTuned", check_pr=True)
tokenizer.push_to_hub("ar111/Llama-2-7b-chat-hf-orcaTuned",check_pr=True)

