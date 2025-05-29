# Full.sh
#!/bin/bash

# Ejecutar el entrenamiento en segundo plano y guardar log
nohup /home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/nohup/run.sh > /home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/logs/salida.log 2>&1

# Ejecutar el Bot en segundo plano y guardar log
nohup /home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/nohup/runBot.sh > /home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/logs/salidaBot.log 2>&1
