# Emotional-State-Tracker

Memoria emocional **offline**, **determinista** y **audit-able** (hash).  
Objetivo: convertir texto a un vector fijo, clasificar un estado emocional por distancia a centroides, y recuperar estado previo por similitud.

## Requisitos (mínimos)
- Python 3.12
- numpy
- scipy
- pytest
- ruff

## Instalación (dev)
```bash
python -m pip install -e ".[ci]"