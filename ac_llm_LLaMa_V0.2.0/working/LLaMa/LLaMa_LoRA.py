from datasets import Dataset
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



########## 
# model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Modelo base 

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

path = "/home/estudiante1/workspaces/ac_llm/dataset/oscar_dataset.xlsx"
output_dir="../model/llama3.1-lora-spanish"
logging_dir="../logs/LLaMa3logs"
train_epochs = 1

########## 


# 1. Cargar el tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Modelo base
# model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Modelo base 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

# 1. Descargar y preparar los datos
# Cargar datos
df = pd.read_excel(path)
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

df = df.head(15000)

dataset = Dataset.from_pandas(df)

# 3. Preprocesar los datos en español
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128  # Reducido para ahorrar memoria
    )

# Aplicar la tokenización al dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=6,
    remove_columns=dataset.column_names  # Eliminar columnas originales
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
    use_auth_token=True,
    quantization_config=quantization_config
)

# 6. Preparar el modelo para entrenamiento con 4-bit
model = prepare_model_for_kbit_training(model)

# 7. Configurar LoRA con parámetros conservadores para ahorrar memoria
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=6,                   # Reducido para ahorrar memoria
    lora_alpha=16,         # Reducido para ahorrar memoria
    lora_dropout=0.05,     # Reducido para ahorrar memoria
    target_modules=[       # Capas específicas para aplicar LoRA
        "q_proj",
        "v_proj"           # Reducido a solo q_proj y v_proj para ahorrar memoria
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

# 10. Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=output_dir,      
    overwrite_output_dir=True,               
    num_train_epochs=train_epochs,                      
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
    logging_dir=logging_dir,                    
    report_to="tensorboard",                 
)

# 11. Definir el Trainer
train_size = int(0.9 * len(tokenized_datasets))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets.select(range(train_size, len(tokenized_datasets))),
    data_collator=data_collator,
)

# 12. Entrenar el modelo
trainer.train()

# 13. Guardar el modelo adaptado
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Modelo adaptado guardado en {0}".format(output_dir))
