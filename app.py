import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, GPTJForQuestionAnswering
import PyPDF2
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_file, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to create embeddings for text chunks
def create_embeddings(text_chunks):
    """Creates embeddings for the text chunks using a pre-trained model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    return embeddings

# Global model and tokenizer for question answering
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
model = GPTJForQuestionAnswering.from_pretrained("hf-internal-testing/tiny-random-gptj")

def answer_question(question, text, max_length=512):
    """Queries the GPT-J model to get the answer to the question."""
    inputs = tokenizer(question, text, return_tensors="pt", truncation=True, max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    # Validate the answer range
    if answer_start_index < answer_end_index and answer_start_index < len(inputs.input_ids[0]) and answer_end_index < len(inputs.input_ids[0]):
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        return answer.strip()
    else:
        return ""  # Return an empty string if no valid answer is found

# Function to chunk text into smaller parts for processing
def chunk_text(text, chunk_size=500):
    """Chunks the text into smaller parts for easier processing."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
        else:
            current_chunk += sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@app.route('/ask', methods=['POST'])
def ask_question_route():
    question = request.form.get('question')
    pdf_file = request.files['pdf_file']

    # Save the uploaded PDF file temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Chunk text for processing
    text_chunks = chunk_text(text)

    # Answer the question for each text chunk and store the answers with their source
    answers = []
    for chunk in text_chunks:
        answer = answer_question(question, chunk)
        if answer:  # Only store non-empty answers
            answers.append((answer, chunk))

    # If there are answers, return the first one with its source
    if answers:
        answer, source_chunk = answers[0]
    else:
        answer = "Sorry, I couldn't find an answer to your question...."
        source_chunk = ""  # No source chunk available

    # Structure the output
    structured_output = {
        "question": question,
        "answer": answer,
        "source_chunk": source_chunk
    }

    return jsonify(structured_output)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
