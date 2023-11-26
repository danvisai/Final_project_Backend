from flask import Flask, render_template, request, send_file
import base64
import re
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from generator import generateImage

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
    #image_file = request.files['image']
    data_url = request.values['image']
    offset = data_url.index(',')
    img_bytes = base64.b64decode(data_url[offset:])
    img = BytesIO(img_bytes)

    data = generateImage(text_input, "low detail, bad quality, blurry" ,img)

    img = BytesIO()
    data.save(img, format="JPEG")
    # img = Image.open(BytesIO(img_bytes))
    # img.show()
    return send_file(img, mimetype='image/png') 

if __name__ == '__main__':
    app.run(debug=True)
