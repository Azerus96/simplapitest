import os
import gradio as gr
from openai import OpenAI
from anthropic import Anthropic

def get_available_models():
    """Получаем доступные модели для обоих провайдеров"""
    models = {
        'OpenAI': [],
        'Anthropic': []
    }
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        openai_models = client.models.list().data
        models['OpenAI'] = [m.id for m in openai_models if 'gpt' in m.id]
    except Exception as e:
        print(f"OpenAI error: {str(e)}")
    
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        models['Anthropic'] = [
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-2.1'
        ]
    except Exception as e:
        print(f"Anthropic error: {str(e)}")
    
    return models

def test_chat(message, provider, model):
    """Тестируем чат с выбранным провайдером"""
    try:
        if provider == "OpenAI":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_tokens=100
            )
            return response.choices[0].message.content
            
        elif provider == "Anthropic":
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Создаем интерфейс при загрузке
available_models = get_available_models()

with gr.Blocks(title="API Keys Tester") as demo:
    gr.Markdown("## 🤖 API Keys Testing Interface")
    
    with gr.Row():
        provider = gr.Dropdown(
            choices=["OpenAI", "Anthropic"],
            label="Select Provider"
        )
        model = gr.Dropdown(
            choices=available_models['OpenAI'] + available_models['Anthropic'],
            label="Select Model"
        )
    
    msg = gr.Textbox(label="Your Message", value="Hello! How are you?")
    btn = gr.Button("Test Chat")
    output = gr.Textbox(label="Response", interactive=False)
    
    btn.click(
        fn=test_chat,
        inputs=[msg, provider, model],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
