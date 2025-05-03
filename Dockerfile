FROM pytorch/pytorch:latest

# Instalação de dependências básicas
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar transformers e outras bibliotecas necessárias
RUN pip install transformers==4.30.2 accelerate sentencepiece

# Criar diretório para armazenar modelos e dados dos usuários
RUN mkdir -p /runpod-volume/user-data

# Copiar o arquivo handler
COPY handler.py /

# Configurar ambiente
ENV PYTHONPATH=/

# Pré-baixar modelo base para acelerar
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('gpt2'); model = AutoModelForCausalLM.from_pretrained('gpt2')"

# Definir entrypoint
ENTRYPOINT ["python", "-m", "handler"]