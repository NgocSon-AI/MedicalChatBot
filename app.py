from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeSparseVectorStore, PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

OPENAI_API_KEY = os.environ.get('OPEN_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ['OPEN_API_KEY'] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "nsmedicalbot"



# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = OpenAI(temperature=0.4, max_tokens=500, api_key=OPENAI_API_KEY)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")  # file HTML chứa giao diện chat

@app.route("/get", methods=["POST"])
def chat():
    try:
        # Nếu dùng JavaScript fetch với JSON:
        data = request.get_json()
        msg = data.get("msg", "")
        
        if not msg:
            return jsonify({"answer": "Sorry, I didn't get your message."}), 400

        print("User input:", msg)
        response = rag_chain.invoke({"input": msg})
        print("Response:", response["answer"])

        return jsonify({"answer": response["answer"]})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"answer": "Sorry, an error occurred."}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)