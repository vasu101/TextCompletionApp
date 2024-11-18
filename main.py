from flask import Flask, request, jsonify, render_template
import requests
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Hugging Face API interaction
def query(payload, model_id="gpt2", api_token=os.getenv("HUGGINGFACE_API_KEY")):
    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# Home route for rendering the frontend
@app.route('/')
def home():
    return render_template('index.html')

# Generate text from the model
@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.form
    prompt = data.get('prompt', '')

    # Call Hugging Face API
    output = query({"inputs": prompt})
    generated_text = output[0]["generated_text"] if "generated_text" in output[0] else "Error in generation"

    return jsonify({"generated_text": generated_text})

# Dynamic chat endpoint
@app.route('/chat')
def call_huggingface_chat_model():
    model_id = request.args.get("model_id", "gpt2")
    huggingface_token = request.args.get("huggingface_token", os.getenv("HUGGINGFACE_API_KEY"))
    questions = request.args.get("input", "")
    data = query(
        {
            "inputs": questions.replace("_", " "),
            "options": {"wait_for_model": True},
            "parameters": {"return_full_text": False, "max_time": 30},
        },
        model_id,
        huggingface_token,
    )
    logging.debug(f"Model output: {data}")
    output = jsonify({"ack": data[0]["generated_text"] if "generated_text" in data[0] else "Error in generation"})
    output.headers.add("Access-Control-Allow-Origin", "*")
    return output

# Load model for readiness
@app.route('/load_model')
def load_huggingface_chat_model():
    model_id = request.args.get("model_id", "gpt2")
    huggingface_token = request.args.get("huggingface_token", os.getenv("HUGGINGFACE_API_KEY"))
    data = query(
        {
            "inputs": "hello",
            "parameters": {"return_full_text": True},
        },
        model_id,
        huggingface_token,
    )
    logging.debug(f"Model output: {data}")
    if "generated_text" in data[0]:
        output = jsonify({"ack": "model loaded and ready"})
    else:
        output = jsonify({"ack": "Error loading model"})
    output.headers.add("Access-Control-Allow-Origin", "*")
    return output

# Health check endpoint
@app.route('/ping')
def ping():
    output = jsonify({"ack": "pong"})
    output.headers.add("Access-Control-Allow-Origin", "*")
    return output

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
