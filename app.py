from flask import Flask, render_template, request
import base64

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
