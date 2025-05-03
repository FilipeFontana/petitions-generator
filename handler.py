import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any

# Configuração do modelo
MODEL_ID = "neuralmind/sabia-2-lawbr"  # Substitua pelo modelo que você deseja usar
USER_DATA_PATH = "/runpod-volume/user-data"  # Onde os dados específicos do usuário serão armazenados

def load_model():
    """Carrega o modelo base"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tokenizer

def get_user_model_path(user_id: str):
    """Obtém o caminho para o modelo específico do usuário"""
    return os.path.join(USER_DATA_PATH, f"user_{user_id}")

def save_documents(user_id: str, documents: List[str]):
    """Salva os documentos do usuário para treinamento"""
    user_dir = get_user_model_path(user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    for i, doc in enumerate(documents):
        with open(os.path.join(user_dir, f"doc_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(doc)
    
    return user_dir

def train_model_for_user(user_id: str, documents: List[str]):
    """Treina o modelo para um usuário específico"""
    model, tokenizer = load_model()
    user_dir = save_documents(user_id, documents)
    
    # Aqui você implementaria o código de fine-tuning
    # Exemplo simplificado:
    # 1. Preparar dados de treinamento
    train_texts = []
    for doc_path in os.listdir(user_dir):
        if doc_path.endswith(".txt"):
            with open(os.path.join(user_dir, doc_path), "r", encoding="utf-8") as f:
                train_texts.append(f.read())
    
    # 2. Tokenizar textos
    train_encodings = tokenizer("\n\n".join(train_texts), truncation=True, padding=True, return_tensors="pt")
    
    # 3. Fine-tuning (simplificado)
    # Na implementação real, você usaria uma abordagem de treinamento adequada
    # como o Trainer do Hugging Face
    
    # 4. Salvar modelo específico do usuário
    model_save_path = os.path.join(user_dir, "model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    return {"status": "completed", "message": "Modelo treinado com sucesso"}

def generate_petition(user_id: str, petition_data: Dict[str, Any]):
    """Gera uma petição usando o modelo treinado do usuário"""
    model_path = os.path.join(get_user_model_path(user_id), "model")
    
    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        return {"error": "Modelo não encontrado. O usuário precisa treinar o modelo primeiro."}
    
    # Carregar modelo específico do usuário
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Construir prompt para a geração
    court = petition_data.get("court", "")
    jurisdiction = petition_data.get("jurisdiction", "")
    process_number = petition_data.get("processNumber", "")
    action_type = petition_data.get("actionType", "")
    
    # Dados do réu
    defendant = petition_data.get("defendantData", {})
    defendant_name = defendant.get("name", "")
    defendant_cpf = defendant.get("cpf", "")
    defendant_rg = defendant.get("rg", "")
    
    address = defendant.get("address", {})
    defendant_address = f"{address.get('street', '')}, {address.get('number', '')}, {address.get('neighborhood', '')}, {address.get('city', '')}, {address.get('state', '')}, {address.get('zipCode', '')}"
    
    # Fatos e pedidos
    facts = petition_data.get("facts", "")
    requests = petition_data.get("requests", "")
    
    # Construir prompt
    prompt = f"""
    EXCELENTÍSSIMO(A) SENHOR(A) DOUTOR(A) JUIZ(A) DE DIREITO DA {court} DA COMARCA DE {jurisdiction}
    
    PROCESSO Nº {process_number}
    
    AÇÃO: {action_type}
    
    [NOME DO AUTOR], já qualificado nos autos do processo em epígrafe, vem, respeitosamente, através de seu advogado que esta subscreve, à presença de Vossa Excelência, apresentar [TIPO DE MANIFESTAÇÃO] em face de {defendant_name}, [qualificação], pelos fatos e fundamentos a seguir expostos:
    
    DOS FATOS
    
    {facts}
    
    DOS PEDIDOS
    
    {requests}
    """
    
    # Gerar texto
    inputs = tokenizer(prompt, return_tensors="pt")
    output_sequences = model.generate(
        inputs['input_ids'],
        max_length=1500,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True
    )
    
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return {"petition_text": generated_text}

def handler(event):
    """Handler principal do endpoint serverless"""
    try:
        input_data = event["input"]
        task = input_data.get("task")
        user_id = input_data.get("userId")
        
        if not user_id:
            return {"error": "ID do usuário não fornecido"}
        
        if task == "train":
            documents = input_data.get("documents", [])
            if not documents:
                return {"error": "Nenhum documento fornecido para treinamento"}
            
            result = train_model_for_user(user_id, documents)
            return result
        
        elif task == "generate":
            petition_data = input_data.get("petitionData", {})
            if not petition_data:
                return {"error": "Dados da petição não fornecidos"}
            
            result = generate_petition(user_id, petition_data)
            return result
        
        else:
            return {"error": f"Tarefa desconhecida: {task}"}
            
    except Exception as e:
        return {"error": str(e)}
