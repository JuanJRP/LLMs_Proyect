import torch
from datasets import load_dataset, Dataset
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  TrainingArguments,
  Trainer,
  DataCollatorForLanguageModeling,
  BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import kagglehub
import pandas as pd
import os

# 1. Cargar el tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Modelo base
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

# 1. Descargar y preparar los datos
path = kagglehub.dataset_download("camiloemartinez/productos-consumo-masivo")

# Cargar datos
df = pd.read_excel(os.path.join(path, "output - Kaggle.xlsx"))
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

selected_cols = ['prod_name_long', 'subcategory', 'tags', 'prod_unit_price']

# Extraer solo las columnas seleccionadas
df = df[selected_cols].copy()

df = df.head(8000)

# Convertir precio a float
df['prod_unit_price'] = df['prod_unit_price'].astype(str).str.replace(',', '.', regex=False).astype(float)

# Crear un ID autogenerado
df['id'] = pd.Series(range(1, len(df) + 1))

# Crear columna text que contenga la información de todas las columnas seleccionadas
df['text'] = df.apply(
    lambda row: f"Nombre: {row['prod_name_long']}, Subcategoría: {row['subcategory']}, "
               f"Tags: {row['tags']}, Precio: {row['prod_unit_price']}", axis=1
)

df = df.drop(selected_cols, axis=1)

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
model.save_pretrained("./llama2-lora-spanish-final")
tokenizer.save_pretrained("./llama2-lora-spanish-final")

print("Modelo adaptado guardado en ./llama2-lora-spanish-final")