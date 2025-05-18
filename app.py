from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import base64
import tensorflow as tf
from PIL import Image

labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]

app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path="model/bisindo_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image_tflite(tflite_path, image_path, labels):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dapatkan ukuran input yang diminta model
    input_shape = input_details[0]['shape']
    img_width, img_height = input_shape[2], input_shape[1]

    img = Image.open(image_path).resize((img_width, img_height))
    input_data = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_index = np.argmax(output)
    return labels[pred_index], float(output[0][pred_index])

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"result": "No image provided"}), 400

    # Decode base64 dan simpan ke file sementara
    image_bytes = base64.b64decode(image_b64)
    image_path = "input_temp.jpg"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Prediksi
    try:
        result, confidence = predict_image_tflite("bisindo_model.tflite", image_path, labels)
        return jsonify({
            "result": result,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

