from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Page numbering starts at 0
        page_text = page.get_text("text")  # Extract text
        pages_text.append(page_text)

    return pages_text

app = Flask(__name__)
CORS(app)

# Configure Google API Key and Model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety settings as per your code
safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=MODEL_CONFIG,
    safety_settings=safety_settings
)

from pathlib import Path

def image_format(image_path):
    img = Path(image_path)

    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "image/png", ## Mime type are PNG - image/png. JPEG - image/jpeg. WEBP - image/webp
            "data": img.read_bytes()
        }
    ]
    return image_parts


def process_image(image_path, system_prompt, user_prompt):
    image_info = image_format(image_path)
    input_prompt = [system_prompt, image_info[0], user_prompt]
    response = model.generate_content(input_prompt)
    return response.text

def process_pdf(pdf_path, system_prompt, user_prompt):
    pages_text = extract_text_from_pdf(pdf_path)
    responses = []
    for page_number, page_text in enumerate(pages_text, start=1):
        input_prompt = [system_prompt, page_text, user_prompt]
        response = model.generate_content(input_prompt)
        responses.append({"page_number": page_number, "response": response.text})
    return responses


# API route to handle the image and PDF file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    system_prompt = request.form['system_prompt']
    user_prompt = request.form['user_prompt']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        if filename.endswith('.pdf'):
            responses = process_pdf(file_path, system_prompt, user_prompt)
        else:
            responses = process_image(file_path, system_prompt, user_prompt)

        return jsonify(responses), 200

    return jsonify({"error": "Invalid file format!"}), 400

def allowed_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))

if __name__ == '__main__':
    app.run(debug=True)
