import gradio as gr
import os
import torch
import torch.nn.functional as F

from model.predict import model, transform, classes

def predict(img):
    img = transform(img).unsqueeze(0).to("cpu")

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)[0]

    conf, pred = torch.max(probs, dim=0)
    confidence = conf.item()

    if confidence < 0.65:
        return f"Uncertain 🤔 (confidence: {confidence:.2f})"

    return f"{classes[pred]} (confidence: {confidence:.2f})"


with gr.Blocks(title="Dog vs Cat Classifier") as demo:
    gr.Markdown("## 🐶🐱 Dog vs Cat Classifier")

    # INPUT AT TOP
    image_input = gr.Image(
        sources=["upload", "webcam"],
        type="pil",
        label="Upload Image or Use Webcam"
    )

    # OUTPUT AT BOTTOM
    result_output = gr.Textbox(
        label="Prediction",
        interactive=False
    )

    classify_btn = gr.Button("Classify")

    classify_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=result_output
    )

if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
