# Full.sh
#!/bin/bash

# Ejecutar el entrenamiento en segundo plano y guardar log
nohup /home/estudiante1/workspaces/ac_llm/working/nohup/run.sh > ../logs/salida.log 2>&1

# Ejecutar el Bot en segundo plano y guardar log
nohup /home/estudiante1/workspaces/ac_llm/working/nohup/runBot.sh > ../logs/salidaBot.log 2>&1
