import gradio as gr
from model.predict import predict_toxicity


def classify_text(text):
    result = predict_toxicity(text)
    label = result['label']
    score = result['score']
    
    if label == 'toxic':
        return f"Текст содержит оскорбительный характер с вероятностью: {score*100:.1f}%"
    else:
        return f"Текст содержит нейтральный характер с вероятностью: {score*100:.1f}%"

demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для проверки", label="Текст"),
    outputs=gr.Textbox(label="Результат"),
    title="Классификатор токсичных комментариев",
    description="Определяет токсичность текста на русском языке",
    examples=[
        "Спасибо за помощь!",
        "Ты полный идиот",
        "Отличная работа",
        "Закрой рот, дебил"
    ]
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)