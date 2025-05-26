# LLaMA LoRA Fine-tuning Project

Este proyecto implementa el ajuste fino (fine-tuning) de modelos LLaMA-2 en español utilizando la técnica LoRA (Low-Rank Adaptation) y la librería PEFT. El objetivo es facilitar el entrenamiento, evaluación y despliegue de modelos de lenguaje adaptados a tareas específicas en español.

## Estructura del Proyecto

```
├── .gitignore
├── LICENSE
├── requirements.txt
├── Commands.txt
├── optimized/
│   ├── LLaMa/
│   ├── logs/
│   ├── nohup/
│   │   ├── llama2-lora-spanish/
│   │   ├── llama2-lora-spanish-final/
│   │   ├── logs/
│   │   ├── run.sh
│   │   └── runBot.sh
│   └── source/
├── working/
│   ├── LLaMa/
│   ├── logs/
│   ├── model/
│   │   └── llama2-lora-spanish/
│   ├── nohup/
│   │   ├── Full.sh
│   │   ├── run.sh
│   │   └── runBot.sh
│   └── source/
```

## 🚀 Características
- **optimized/**: Scripts y modelos optimizados para entrenamiento y despliegue.
- **working/**: Área de trabajo principal para experimentación, logs y checkpoints.
- **model/**: Checkpoints y resultados de modelos entrenados.
- **nohup/**: Scripts de ejecución en segundo plano para entrenamiento y bots.
- **logs/**: Salidas y registros de entrenamiento y bots.
- **source/**: Scripts de prueba y utilidades.


## 💻 Requisitos del Sistema

### Hardware Mínimo

- **GPU**: NVIDIA con al menos 8GB VRAM (recomendado 16GB+)
- **RAM**: 16GB mínimo (recomendado 32GB+)
- **Almacenamiento**: 50GB de espacio libre


### Software

- Python 3.8+
- CUDA 11.8+ (compatible con PyTorch 2.0+)w
- Git


## Instalación

1. Clona el repositorio.
```bash
git clone https://github.com/JuanJRP/LLaMa_LoRA_Project.git
cd LLaMa-LoRA-Project
```
### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```
### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```
### 4. Configurar Hugging Face
```bash
huggingface-cli login
```
Necesitarás un token de Hugging Face con acceso al modelo LLaMA-2.

## ⚙️ Configuración

### 1. Acceso al modelo LLaMA-2
- Solicita acceso a LLaMA-2 en [Meta AI](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- Acepta los términos en [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
### 2. Token de Telegram Bot
- Crea un bot con [@BotFather](https://t.me/botfather)
- Reemplaza `TELEGRAM_TOKEN` en `Test_Model.py` con tu token
```python
TELEGRAM_TOKEN = "tu_token_aqui"
```
### 3. Configuración de paths
Actualiza las rutas en los scripts según tu estructura de directorios:
```bash
# En run.sh y runBot.sh
python /ruta/a/tu/proyecto/LLaMa_LoRA.py
```

## Entrenamiento y Ejecución
Para entrenar el modelo y ejecutar el bot, utiliza los scripts en `working/nohup/`:
```sh
bash working/nohup/Full.sh
```
Esto ejecutará el entrenamiento y el bot en segundo plano, guardando los logs en `working/logs/`.

## 🔧 Detalles Técnicos

### Configuración de LoRA

```python
LoraConfig(
    r=6,                    # Rango de matrices de bajo rango
    lora_alpha=16,          # Factor de escala
    lora_dropout=0.05,      # Dropout para regularización
    target_modules=["q_proj", "v_proj"]  # Capas objetivo
)
```

### Optimizaciones de Memoria
- **Cuantización 4-bit**: Reduce uso de memoria en ~75%
- **Gradient Checkpointing**: Intercambia memoria por cómputo
- **Batch Size Pequeño**: 2 ejemplos por dispositivo
- **Gradient Accumulation**: 16 pasos para batch efectivo de 32

### Parámetros de Entrenamiento
```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    fp16=True
)
```

### Dataset

- **Fuente**: Kaggle - Productos de Consumo Masivo
- **Tamaño**: 8,000 productos (limitado para pruebas)
- **Formato**: Texto estructurado con nombre, subcategoría, tags y precio
- **División**: 90% entrenamiento, 10% evaluación

## 🛠️ Solución de Problemas

### Error de Memoria CUDA
```
RuntimeError: CUDA out of memory
```

**Solución:**
- Reduce `per_device_train_batch_size` a 1
- Aumenta `gradient_accumulation_steps`
- Reduce `max_length` en tokenización

### Token de Hugging Face
```
401 Client Error: Unauthorized
```

**Solución:**
```bash
huggingface-cli login --token your_token_here
```
### Error de Permisos en Scripts
```bash
chmod +x run.sh runBot.sh

```

### Bot de Telegram no responde
1. Verifica que el token sea correcto
2. Confirma que el modelo esté cargado correctamente
3. Revisa los logs de errores en consola

### Instalación de BitsAndBytesConfig
Si encuentras errores con `bitsandbytes`:
```bash
# Para sistemas con CUDA
pip install bitsandbytes
# Para sistemas sin CUDA
pip install bitsandbytes-cpu
```
## 📊 Métricas y Evaluación

### Parámetros Entrenables
El modelo muestra qué porcentaje de parámetros se entrena:
```
trainable params: 1,769,472 || all params: 6,740,781,056 || trainable%: 0.02625
```

### Métricas de TensorBoard
- Loss de entrenamiento
- Learning rate schedule
- Gradient norms
- Tiempo por step

## 📝 Notas Importantes

### Consideraciones de Licencia
- LLaMA-2 requiere licencia de Meta AI
- Uso comercial limitado según términos de Meta
- Revisar términos antes de producción

### Rendimiento
- Tiempo de entrenamiento: ~2-4 horas en GPU moderna
- Inferencia: ~1-3 segundos por respuesta
- Tamaño del modelo: ~13GB (modelo base + adaptadores LoRA)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia
Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 📞 Soporte

Para preguntas o problemas:
- Abre un issue en GitHub
- Consulta la documentación de [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- Revisa la documentación de [PEFT](https://huggingface.co/docs/peft)

---

**⚠️ Advertencia**: Este es un proyecto educativo. Para uso en producción, considera aspectos adicionales de seguridad, escalabilidad y cumplimiento normativo.