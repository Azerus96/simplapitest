from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from anthropic import Anthropic
import os
import requests
import logging

app = Flask(__name__)

# Конфигурация провайдеров
MINIMAX_API_URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
MINIMAX_MODELS = [
    "MiniMax-Text-01",
    "abab6.5s-chat",
    "DeepSeek-R1"
]

# Настройка логгера
logging.basicConfig(level=logging.INFO)

def get_models(provider: str) -> list:
    """Получаем доступные модели для выбранного провайдера"""
    try:
        if provider == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return [model.id for model in client.models.list().data if 'gpt' in model.id]
            
        elif provider == "anthropic":
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            return [
                'claude-3-5-sonnet-20240620',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-2.1'
            ]
            
        elif provider == "minimax":
            return MINIMAX_MODELS
            
    except Exception as e:
        app.logger.error(f"Error getting models: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_models', methods=['POST'])
def handle_get_models():
    provider = request.json.get('provider')
    return jsonify({"models": get_models(provider)})

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    provider = data.get('provider')
    
    try:
        if provider == "openai":
            return handle_openai_chat(data)
        elif provider == "anthropic":
            return handle_anthropic_chat(data)
        elif provider == "minimax":
            return handle_minimax_chat(data)
        else:
            return jsonify({"error": "Invalid provider"}), 400
            
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def handle_openai_chat(data):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=data['model'],
        messages=[{"role": "user", "content": data['message']}],
        max_tokens=100
    )
    return jsonify({"response": response.choices[0].message.content})

def handle_anthropic_chat(data):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=data['model'],
        max_tokens=100,
        messages=[{"role": "user", "content": data['message']}]
    )
    return jsonify({"response": response.content[0].text})

def handle_minimax_chat(data):
    headers = {
        "Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": data['model'],
        "messages": [{"role": "user", "content": data['message']}],
        "temperature": 0.1,
        "max_tokens": 256
    }
    
    response = requests.post(MINIMAX_API_URL, json=payload, headers=headers)
    response_data = response.json()
    
    if response.status_code == 200 and response_data['base_resp']['status_code'] == 0:
        return jsonify({"response": response_data['choices'][0]['message']['content']})
    else:
        error_msg = response_data.get('base_resp', {}).get('status_msg', 'Unknown error')
        raise Exception(f"MiniMax API Error: {error_msg}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
