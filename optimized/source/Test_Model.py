from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

import torch
import asyncio
import nest_asyncio

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
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

peft_model_id = "/home/estudiante1/workspaces/ac_llm/optimized/nohup/llama2-lora-spanish-final"
testModel, testTokenizer = test_model(peft_model_id)

# ---------- Telegram bot ----------
TELEGRAM_TOKEN = "YOUR TOKEN HERE"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text
    print(prompt)
    inputs = testTokenizer(prompt, return_tensors="pt").to(testModel.device)
    with torch.no_grad():
        outputs = testModel.generate(
            **inputs,
            eos_token_id=testTokenizer.eos_token_id,
            max_new_tokens=300,
            temperature=0.9,
            top_p=0.9,
            do_sample=True
        )
    response = testTokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    if prompt in response:
        response = response.replace(prompt, "").strip()
        print(response)

    await update.message.reply_text(response)

# Iniciar aplicación de Telegram
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.run_polling()

nest_asyncio.apply()

if __name__ == "__main__":
    asyncio.run(main())