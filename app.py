import os
import gc
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from trl import SFTTrainer, SFTConfig

# Ensure Git is installed before pushing to Hugging Face

# Model and Dataset Configuration
model_name = "abhinand/tamil-llama-7b-base-v0.1"
dataset_name = "girishnP/TAMILI"
instruction_column = "text"
new_model = "girishnP/tamil-llama-custom"
local_model_dir = "./model-hack"  # Local model save directory
use_flash_attention_2 = False  # Ensure compatibility with PyTorch version

# LoRA Configuration
lora_r = 64
lora_alpha = 128
lora_dropout = 0.1
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "down_proj", "up_proj",
]

# BitsAndBytes Configuration (Optional)
use_4bit = False
use_8bit = False
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Training Arguments
output_dir = "./results"
num_train_epochs = 2
fp16 = True  # Enable mixed-precision training
bf16 = False
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
gradient_accumulation_steps = 8
gradient_checkpointing = True
max_grad_norm = 1.0
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 500
logging_steps = 100
eval_strategy = "steps"
eval_steps = 500

# Load and Preprocess Dataset
print("🔄 Loading Dataset...")
dataset = load_dataset(dataset_name, split="train")

# Ensure "Genre" column exists before concatenation
if "Genre" in dataset.column_names:
    dataset = dataset.map(lambda x: {"text": f"[{x['Genre']}] {x['text']}"})
else:
    print("⚠️ 'Genre' column not found, proceeding with 'text' only.")

# Split into train/eval datasets
train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size)).shuffle(seed=42)
eval_dataset = dataset.select(range(train_size, len(dataset)))

print(f"✅ Sample Instruction: {train_dataset[0][instruction_column]}\n")

# Quantization Configuration (if needed)
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load Model
print("🚀 Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    use_flash_attention_2=use_flash_attention_2,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_target_modules,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    resume_from_checkpoint=True,  # Resume training if checkpoint exists
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    eval_steps=eval_steps,
    evaluation_strategy=eval_strategy,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments
)

print("🎯 Starting Training...")

# Train and Evaluate
trainer.train()
eval_results = trainer.evaluate()
print(f"✅ Evaluation results: {eval_results}")

# Save Trained Model
print("💾 Saving Model Locally...")
os.makedirs(local_model_dir, exist_ok=True)
trainer.model.save_pretrained(local_model_dir)
tokenizer.save_pretrained(local_model_dir)
print(f"✅ Model saved locally at: {local_model_dir}")

# Clean Up Memory
del model
del trainer
gc.collect()
torch.cuda.empty_cache()

# Load and Merge Model for Pushing to Hugging Face Hub
print("🔄 Merging Trained Weights with Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, local_model_dir)
model = model.merge_and_unload()

# Reload Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Push to Hugging Face Hub
print("📤 Pushing Model and Tokenizer to Hugging Face...")
try:
    model.push_to_hub(new_model, use_temp_dir=False, private=True)
    tokenizer.push_to_hub(new_model, use_temp_dir=False, private=True)
    print("✅ Model and Tokenizer pushed to Hugging Face Hub successfully!")
except Exception as e:
    print(f"❌ Error pushing to Hugging Face: {e}")


