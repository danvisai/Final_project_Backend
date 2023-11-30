from flask import Flask, render_template, request, send_file
import base64
import re
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from generator import generateImage
from torchvision import transforms
import io

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        text_input = request.form['text']
        image_file = request.files['image']
        image_data = image_file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        return render_template('result.html', text=text_input, image_data=encoded_image)
    
@app.route('/getImage', methods=['POST'])

def getImage():
    text_input = request.form['prompt']
    img_str = request.values['image']
    print("image str data",img_str)
    image= None
    if img_str.startswith('data:image'):
        
        # Find the start of the base64 string
        img_str_offset = img_str.find('base64,') + len('base64,')
        print(img_str)
        # Grab the actual base64 content (after the comma)
        img_str = img_str[img_str_offset:]
        
        img_bytes = base64.b64decode(img_str)

        img_buf = io.BytesIO(img_bytes)


    # Byte stream to PIL Image
        image = Image.open(img_buf).convert('RGB')
    
    # Assuming generateImage is a function to create an image based on text_input
    data = generateImage(text_input, "low detail, bad quality, blurry", image=image)
    
    # For testing purposes, opening a sample image file
    #data = Image.open("./input/test.jpeg")

    # Saving the image data into BytesIO object
    image=transforms.ToPILImage()(data.squeeze().cpu())
    img_io = BytesIO()
    image.save(img_io, format='JPEG')
    img_io.seek(0)  # Move cursor to the start of the BytesIO stream
    
    # Return the image file using Flask's send_file function
    return send_file(img_io, mimetype='image/jpeg')

    

if __name__ == '__main__':
    app.run(debug=True)
