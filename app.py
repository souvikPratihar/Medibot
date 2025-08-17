import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Load Groq API key
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

@st.cache_resource
def create_qa_chain():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-8b-8192",      # or "llama3-8b-8192", "gemma-7b-it"
        temperature=0.7,
        max_tokens=512,
    )

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
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def main():
    st.set_page_config(page_title="Medical Chatbot 🩺", page_icon="🩺")
    st.title("Medical Chatbot 🩺")
    st.caption("Your personal medical encyclopedia assistant")

    qa_chain = create_qa_chain()

    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about a medical condition..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
            with st.chat_message("assistant"):
                st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()

