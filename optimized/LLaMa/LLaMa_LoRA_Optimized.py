import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import kagglehub
import pandas as pd

# 1. Tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Datos (usando solo una pequeña parte para no saturar GPU)
path = kagglehub.dataset_download("camiloemartinez/productos-consumo-masivo")
dataset = load_dataset("oscar", "unshuffled_deduplicated_es", streaming=True)
small_dataset = []
for i, example in enumerate(dataset["train"]):
    if i >= 8000:
        break
    small_dataset.append(example)
dataset = Dataset.from_list(small_dataset)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128 
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,  
    remove_columns=dataset.column_names
)

# 3. Cargar modelo en float16 sin cuantización
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,     
    device_map="auto",            
    use_auth_token=True
)

# 4. Configuración LoRA ligera
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=6,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

# 5. Aplicar LoRA
model.gradient_checkpointing_enable()  
model.enable_input_require_grads() 
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 6. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. Training args
training_args = TrainingArguments(
    output_dir="./llama2-lora-spanish",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    fp16=True,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    logging_dir="./logs",
    report_to="tensorboard",
)

# 8. Trainer
train_size = int(0.9 * len(tokenized_datasets))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets.select(range(train_size, len(tokenized_datasets))),
    data_collator=data_collator,
)

# 9. Entrenamiento
trainer.train()

# 10. Guardar
model.save_pretrained("./llama2-lora-spanish-final")
tokenizer.save_pretrained("./llama2-lora-spanish-final")

print("✅ Modelo adaptado y guardado en ./llama2-lora-spanish-final")
