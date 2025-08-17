import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load Groq API key from environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm():
    llm = ChatOpenAI(
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-8b-8192",   
        temperature=0.7,
        max_tokens=512,
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.
Only if you get greetings message, try to greet also shortly and tell to come to your scope to give answers. 
Don't confuse 'hi' or 'hello', it is normal greetings, so, answer accordingly and shortly, don't get confused, be unambiguous and precise 

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

while True:
    user_query = input("\nWrite Query Here (or type 'exit' to quit): ")
    if user_query.lower() in ["exit", "quit", "tata", "bye", "goodbye", "ok bye"]:
        print("Exiting chatbot. Goodbye! Take care!")
        break
    
    response = qa_chain.invoke({'query': user_query})
    print("\nRESULT: ", response["result"])
    # print("SOURCE DOCUMENTS: ", response["source_documents"])
