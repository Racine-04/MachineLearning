import json
from transformers import pipeline
from flask import Flask, render_template, request
from waitress import serve

app = Flask(__name__)
model = pipeline("image-classification", model="microsoft/resnet-50")

#At first I thought we had to do a UI
@app.route("/", methods=["POST", "GET"])
def index():
    url = request.form.get("link")
    try:
        output = json.dumps(model(url))

    except Exception as error:
        output = error

    return render_template("index.html", result=output)

@app.route("/predict", methods=["POST"])
def predict():
    image = request.data
    image_url = image.decode('utf-8')

    try:
        output = model(image_url)
    except Exception as error:
        output = str(error)
        return output, 415

    return json.dumps(output), 200


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=80)