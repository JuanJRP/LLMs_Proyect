from datasets import Dataset, load_dataset
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  TrainingArguments,
  Trainer,
  DataCollatorForLanguageModeling,
  BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

import torch
import pandas as pd

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

path = "dataset/qna_deepseek_format_106K.jsonl"
output_dir="model/DeepSeek-R1"
logging_dir="logs/DeepSeek-R1-Logs"
train_epochs = 3

# 1. Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. preparar los datos
dataset = load_dataset("json", data_files=path, split="train")

# 3. Preprocesar los datos en español
def tokenize_function(examples):
    input_ids = []
    attention_masks = []

    for messages in examples["messages"]:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        input_ids.append(tokenized)
        attention_masks.append([1] * len(tokenized))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }

# Aplicar la tokenización al dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=6,
    remove_columns=["messages"]
)

print("Dataset tokenizado completado")

# 4. Configurar la cuantización de 4-bit para máximo ahorro de memoria
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 5. Cargar el modelo con cuantización de 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)

# 6. Preparar el modelo para entrenamiento con 4-bit
model = prepare_model_for_kbit_training(model)

# 7. Configurar LoRA con parámetros conservadores para ahorrar memoria
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,                   
    lora_alpha=16,         
    lora_dropout=0.1,     
    target_modules=[       # Capas específicas para aplicar LoRA
        "q_proj",
        "k,proj",
        "v_proj",          
        "0_proj"
    ],
    bias="none",
)

# 8. Aplicar configuración LoRA al modelo
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 9. Configurar el colector de datos para el entrenamiento de LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 10. Definir el Trainer
train_size = int(0.9 * len(tokenized_datasets))
eval_size = len(tokenized_datasets) - train_size

train_dataset = tokenized_datasets.select(range(train_size))
eval_dataset = tokenized_datasets.select(range(train_size, len(tokenized_datasets)))

print("Dataset Dividido: {0} entrenamiento, {1} evaluacion".format(train_size, eval_size))

# 11. Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=output_dir,      
    overwrite_output_dir=True,               
    num_train_epochs=train_epochs,                      
    per_device_train_batch_size=8,           
    per_device_eval_batch_size=8,           
    gradient_accumulation_steps=4,           
    gradient_checkpointing=True,             
    fp16=True,                              
    learning_rate=2e-4,                      
    weight_decay=0.01,                       
    warmup_steps=10,                        
    save_strategy="steps",  
    eval_strategy="steps",           
    eval_steps=100,                          
    save_steps=100,                          
    save_total_limit=2,                      
    logging_steps=100,                        
    logging_dir=logging_dir,                    
    report_to="tensorboard",   
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_drop_last=False,
    remove_unused_columns=False,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 12. Entrenar el modelo
trainer.train()

# 13. Guardar el modelo adaptado
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Modelo adaptado guardado en {0}".format(output_dir))
