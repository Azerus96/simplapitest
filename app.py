from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from anthropic import Anthropic
import os
import logging

app = Flask(__name__)

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
    model = data.get('model')
    message = data.get('message')

    try:
        if provider == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_tokens=100
            )
            return jsonify({"response": response.choices[0].message.content})
            
        elif provider == "anthropic":
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": message}]
            )
            return jsonify({"response": response.content[0].text})
            
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
