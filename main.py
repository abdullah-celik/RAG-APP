import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Initialize Ollama model and other necessary components
MODEL = "llama3"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)
parser = StrOutputParser()

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

chain = None  # Langchain Chain (will be initialized after PDF upload)

# Serve static files from the 'src' directory
@app.route("/")
def index():
    return send_from_directory('src', 'index.html')

# Endpoint to handle PDF upload and initialize Langchain Chain
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
        
        # Initialize Langchain Chain after PDF upload
        loader = PyPDFLoader(filepath)
        pages = loader.load()

        vectorstore = DocArrayInMemorySearch.from_documents(pages, embeddings)
        retriever = vectorstore.as_retriever()

        # Prompt Template for question answering
        template = """
        Answer the question based on the Textbook below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        # Define Langchain Chain
        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | model
            | parser
        )

        return jsonify({"message": "File uploaded successfully and chain initialized."}), 200

# Endpoint to handle questions and interact with the Langchain Chain
@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    response = jsonify({"response here"})
def main():
    app.run(port=int(os.environ.get('PORT', 5000)))

if __name__ == "__main__":
    main()
