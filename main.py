import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the text to image model
model = tf.keras.models.load_model('text_to_image_model.h5')

# Define a function to generate an image from text input
def generate_image(text):
    # Preprocess the text input
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.strip()
    text = ' '.join(text.split()[:20])

    # Generate an image from the text input
    noise = np.random.normal(0, 1, (1, 100))
    condition = np.zeros((1, 10))
    condition[0, 0] = 1
    text_embedding = model.layers[2](model.layers[1](tf.constant([text])))[0]
    generated_image = model.layers[3]([noise, condition, text_embedding])[0]
    generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)

    # Return the generated image
    return Image.fromarray(generated_image)

# Create a Gradio interface for the text to image model
gr.Interface(
    fn=generate_image,
    inputs=gr.inputs.Textbox(label='Input Text'),
    outputs='image',
    title='Text to Image',
    description='Generate an image from text input.'
).launch();
