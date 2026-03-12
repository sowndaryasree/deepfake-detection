import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("deepfake_model.h5")

IMG_SIZE = (224,224)

def predict(img):

    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        return "REAL FACE"
    else:
        return "FAKE FACE"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Deepfake Detection AI"
)

demo.launch(share=True)
