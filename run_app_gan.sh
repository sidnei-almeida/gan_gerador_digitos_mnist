#!/bin/bash

echo "🧠 Iniciando o app GAN - Gerador de Dígitos MNIST..."
echo "=================================================="

if [ ! -d "venv" ]; then
    echo "❌ Ambiente virtual não encontrado!"
    echo "Criando venv..."
    python3 -m venv venv || python -m venv venv || exit 1
fi

echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

echo "📦 Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Iniciando o app Streamlit (http://localhost:8501)"
echo "=================================================="
streamlit run app.py


