import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("deepfake_model.h5")

IMG_SIZE = (224,224)

def predict(img):

    img = img.resize((224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        confidence = pred * 100
        return f"🟢 REAL FACE\nConfidence: {confidence:.2f}%"
    else:
        confidence = (1 - pred) * 100
        return f"🔴 FAKE FACE\nConfidence: {confidence:.2f}%"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Deepfake Detection AI",
    description="Upload a face image to detect whether it is real or AI-generated."
)

demo.launch(share=True)
