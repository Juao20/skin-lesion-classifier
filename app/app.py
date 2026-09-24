"""Démo Gradio."""
import gradio as gr


def predict(image):
    # TODO: charger le modèle et prédire
    return {}


demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=3))

if __name__ == "__main__":
    demo.launch()
