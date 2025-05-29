from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

import torch
import asyncio
import os
import nest_asyncio

# ---------- Formatear prompt para DeepSeek-R1 ----------
def format_deepseek_prompt(user_input, system_prompt):
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def extract_assistant_response(full_response, original_prompt):
    # Remover el prompt original
    if original_prompt in full_response:
        response = full_response.replace(original_prompt, "").strip()
    else:
        response = full_response
    
    # Si hay tokens de fin, cortar ahí
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()
    
    if "</im_end|>" in response:
        response = response.split("</im_end|>")[0].strip()
    
    # Remover tokens no deseados específicos del modelo
    unwanted_tokens = [
        "<\\begin_of_sentence\\>",
        "<begin_of_sentence>", 
        "<\\begin_of_sentece\\>",  
        "<begin_of_sentece>",
        "\\begin_of_sentence\\",
        "begin_of_sentence",
        "<｜begin▁of▁sentence｜>",
        "<｜im_start｜>"
    ]
    
    for token in unwanted_tokens:
        response = response.replace(token, "").strip()
    
    if "<think>" in response and "</think>" in response:
        parts = response.split("</think>")
        if len(parts) > 1:
            final_response = parts[1].strip()
            if final_response:
                for token in unwanted_tokens:
                    final_response = final_response.replace(token, "").strip()
                return final_response
    
    return response

# ---------- Cargar modelo LoRA ----------
def test_model(peft_model_id):
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,   
        device_map="auto",
        token=True
    )
    model.gradient_checkpointing_enable()  
    model.enable_input_require_grads() 
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

peft_model_id = "model/DeepSeek-R1"
testModel, testTokenizer = test_model(peft_model_id)

# ---------- PROMPT DEL SISTEMA  ----------
SYSTEM_PROMPT = """Eres un asistente inteligente y útil que responde en español. 

Características de tus respuestas:
- Responde de manera clara y concisa
- Usa un tono amable y profesional
- Proporciona información precisa y útil
- Si no sabes algo, admítelo honestamente
- Adapta tu nivel de explicación al contexto de la pregunta

Responde siempre en español, sin importar el idioma de la pregunta."""

print("Modelo Cargado")

# ---------- Telegram bot ----------
TELEGRAM_TOKEN = "YOUR_TOKEN"
# TELEGRAM_TOKEN = os.getenv("TOKEN_BOT") ERROR NONE IN SH

async def start_command(update, context):
    await update.message.reply_text("¡Hola! Soy tu DeepSeek-R1 \n\nDesarrollado por:\nJuanJRP - JuliMendez8. \n\n¿En qué puedo ayudarte?")

async def help_command(update, context):
    await update.message.reply_text("Comandos disponibles:\n/start - Iniciar\n/help - Comandos\n/repo - Repositorio de GitHub\n/data - DataSets\n/docs - Documentacion")

async def repo_command(update, context):
    await update.message.reply_text("Puedes encontrar el codigo en: www.github.com/JuanJRP/LLMs_Proyect")

async def data_command(update, context):
    await update.message.reply_text("data url")

async def docs_command(update, context):
    await update.message.reply_text("docs url")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    print(f"Usuario: {user_message}")
    
    formatted_prompt = format_deepseek_prompt(user_message, SYSTEM_PROMPT)
    print(f"Prompt completo generado...")
    
    # Tokenizar
    inputs = testTokenizer(formatted_prompt, return_tensors="pt").to(testModel.device)
    
    try:
        # Generar respuesta
        with torch.no_grad():
            outputs = testModel.generate(
                **inputs,
                eos_token_id=testTokenizer.eos_token_id,
                max_new_tokens=250,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=testTokenizer.pad_token_id,
                repetition_penalty=1.1
            )
        
        # Decodificar respuesta completa
        full_response = testTokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        
        print(f"=== RESPUESTA COMPLETA DEL MODELO ===")
        print(repr(full_response))
        print("="*50)
        
        # Extraer solo la respuesta del asistente
        clean_response = extract_assistant_response(full_response, formatted_prompt)
        
        print(f"Respuesta: {clean_response}")
        
        # Validar que hay respuesta
        if not clean_response or len(clean_response.strip()) < 3:
            clean_response = "Lo siento, no pude generar una respuesta adecuada. ¿Podrías reformular tu pregunta?"
        
        # Enviar respuesta (limitar longitud para Telegram)
        if len(clean_response) > 4000:
            clean_response = clean_response[:3997] + "..."
        
        await update.message.reply_text(clean_response)
        
    except Exception as e:
        print(f"Error generando respuesta: {e}")
        await update.message.reply_text("Ocurrió un error al procesar tu mensaje. Intenta de nuevo.")

# Iniciar aplicación de Telegram
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("repo", repo_command))
    app.add_handler(CommandHandler("data", data_command))
    app.add_handler(CommandHandler("docs", docs_command))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot iniciado con prompt del sistema...")
    print(f"Sistema configurado: {SYSTEM_PROMPT[:100]}...")
    await app.run_polling()

nest_asyncio.apply()

if __name__ == "__main__":
    asyncio.run(main())
