
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import pyttsx3
import os
import uuid
import json
import base64

app = Flask(__name__)
CORS(app)

# Use environment variable for Google API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PROMPT_TEMPLATE = """
Answer the question based only on the following context and provide response with proper formatting to be displayed in a webpage:

{context}

---

Answer the question based on the above context: {question}
"""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    faiss_bytes = vector_store.serialize_to_bytes()
    faiss_base64 = base64.b64encode(faiss_bytes).decode('utf-8')
    faiss_json = {"faiss_index": faiss_base64}
    faiss_bytes = base64.b64decode(faiss_json["faiss_index"])
    vector_store_deserialized = FAISS.deserialize_from_bytes(faiss_bytes, embeddings=embeddings, allow_dangerous_deserialization=True)
    vector_store_deserialized.save_local("faiss_index")
    return json.dumps(faiss_json, indent=4)

def ollama_llm(question, context):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search_with_score(user_question)
    with open("output.txt", 'w', encoding='utf-8') as file:
        for doc, score in docs:
            file.write(f"Content: {doc.page_content}\n")
            file.write(f"Score: {score}\n\n")
    context = " ".join([doc[0].page_content for doc in docs])
    response = ollama_llm(user_question, context)
    return response, context

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    pdf_files = request.files.getlist('pdf_files')
    if not pdf_files:
        return jsonify({"error": "No PDF files uploaded"}), 400
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    return jsonify({"message": "PDF files processed successfully", "embeddings": vector_store}), 200

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    response, context = user_input(question)
    return jsonify({"response": response, "context": context, "session_id": session_id}), 200

@app.route('/new_chat', methods=['POST'])
def new_chat():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id}), 200

if __name__ == '__main__':
    app.run(debug=False)
