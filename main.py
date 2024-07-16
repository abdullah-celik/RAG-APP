import os

from flask import Flask, send_file

app = Flask(__name__)


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from flask import Flask, send_file
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.prompts import PromptTemplate

# RAG Configuration
MODEL = "llama3"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)
parser = StrOutputParser()

# Flask App
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# Langchain Chain (will be initialized after PDF upload)
chain = None 

@app.route("/")
def index():
    return send_file('src/index.html')

# Endpoint to handle PDF upload
@app.route("/upload", methods=["POST"])
def upload_pdf():
    global chain
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    loader = PyPDFLoader(filepath)
    pages = loader.load_

    vectorstore = DocArrayInMemorySearch.from_documents(pages, embeddings)
    retriever = vectorstore.as_retriever()
        # Load the uploaded document and create the chain

    # Prompt Template
    template = """
    Answer the question based on the Textbook below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    # Langchain Chain
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )

    @app.route("/")
    def index():
        return send_file('src/index.html')

    # Endpoint to handle questions (you'll need to adjust this based on how you send questions from your frontend)
    @app.route("/ask", methods=["POST"])
    def ask():
        question = request.form.get("question")  # Assuming question is sent in a form
        response = chain.run(question)
        return jsonify({"answer": response}) 

    def main():
        app.run(port=int(os.environ.get('PORT', 80)))

    if __name__ == "__main__":
        main()
