# from flask import Flask, render_template
from flask import Flask, render_template, request, send_file
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub

app = Flask(__name__)

# Ensure the output directory exists
OUTPUT_DIR = 'static/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the pre-trained TensorFlow Hub model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Load and preprocess images
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/singleStyleTransfer.html')
def single_style_transfer():
    return render_template('singleStyleTransfer.html')

@app.route('/multiStyleTransfer.html')
def multi_style_transfer():
    return render_template('multiStyleTransfer.html')


@app.route('/home2.html')
def home2():
    return render_template('home2.html')

@app.route('/api/style-transfer', methods=['POST'])
def style_transfer():
    # Get the uploaded files
    content_image = request.files['contentImage']
    style_image = request.files['styleImage']

    # Save the uploaded files
    content_path = os.path.join(OUTPUT_DIR, 'content.jpg')
    style_path = os.path.join(OUTPUT_DIR, 'style.jpg')
    content_image.save(content_path)
    style_image.save(style_path)

    # Load the content and style images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Stylize the image
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    output_image = tensor_to_image(stylized_image)
    output_path = os.path.join(OUTPUT_DIR, 'stylized_image.jpg')
    output_image.save(output_path)

    # Return the path to the generated image for downloading
    return {'resultImageUrl': f'/static/output/stylized_image.jpg'}  # Correct URL

# @app.route('/api/multi-style-transfer', methods=['POST'])
# def multi_style_transfer1():
#     # Get the uploaded files
#     content_image = request.files['contentImage']
#     style_images = request.files.getlist('styleImages')  # Get multiple style images

#     # Save the uploaded content image
#     content_path = os.path.join(OUTPUT_DIR, 'content.jpg')
#     content_image.save(content_path)

#     # Load the content image
#     content_image = load_img(content_path)

#     result_image_urls = []

#     # Process each style image
#     for idx, style_image in enumerate(style_images):
#         style_path = os.path.join(OUTPUT_DIR, f'style_{idx}.jpg')  # Unique file name for each style
#         style_image.save(style_path)

#         # Load the style image
#         style_image_tensor = load_img(style_path)

#         # Stylize the image
#         stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image_tensor))[0]
#         output_image = tensor_to_image(stylized_image)
#         output_path = os.path.join(OUTPUT_DIR, f'stylized_image_{idx}.jpg')  # Unique output file name
#         output_image.save(output_path)

#         # Append the URL of the generated image to the list
#         result_image_urls.append(f'/static/output/stylized_image_{idx}.jpg')

#     # Return the paths to the generated images for downloading
#     return {'resultImageUrls': result_image_urls}  # Return a list of result image URLs



if __name__ == '__main__':
    app.run(debug=True)





