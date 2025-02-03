import os
import streamlit as st
import base64
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



DB_custom_path = 'VectorStore/data_base'
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_custom_path, embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_templates):
    prompt = PromptTemplate(template=custom_prompt_templates, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm



def main():
    st.title("Dishy Bot")
    st.markdown("""
    Welcome to Dishy Bot! 🍳🤖smart assistant for step-by-step cooking instructions. Explore recipes like Nargisy Kababs, Chicken Mussallam, and Chicken Shashlik by simply asking with a dish name or ingredients.")

""")

    with st.sidebar:
                # Add social media links
        st.write("# Connect with Me")
        st.write("[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://github.com/engineer12895)")
        st.write("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/junaid-badshah-ai-developer/)")
        
        st.header("User Information")
        user_name = st.text_input("Name")
        user_email = st.text_input("Email")

        if user_name and user_email:
            st.success(f"Welcome, {user_name}!")
            st.info(f"News will be sent to: {user_email}")






    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter your prompt here?")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        custom_prompt_templates = """
                You are an AI assistant that provides step-by-step cooking instructions based on the given context.
                Use the retrieved context below to answer the user's question.
                Context: {context}
                Question: {question}
            Instructions:
                - First, extract and list all required ingredients in a bullet list format. For example:
                 Ingredients:
                - 
                - 
                - 
                - Next, provide the cooking method in details as a numbered list of steps. For example:
            Steps:
                1. 
                2. 
                3. 
            - If the context does not contain enough information to extract both ingredients and steps, reply with: "I don't have enough information to answer this."
            - Do not invent any ingredients or steps—only use the information present in the context.
            """

        huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorestore = get_vectorstore()
            if vectorestore is None:
                st.error("Error in loading vectorstore")
                
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=huggingface_repo_id,HF_TOKEN=HF_TOKEN),
                chain_type = "stuff",
                retriever = vectorestore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs = {'prompt':set_custom_prompt(custom_prompt_templates)}
            )

            response=qa_chain.invoke({'query': prompt})
            result = response["result"]

            source_documents = response.get("source_documents", [])
            formatted_sources = "\n\n".join([doc.page_content for doc in source_documents])
            result_to_show = f"{result}\n\n**Source Docs:**\n{formatted_sources if formatted_sources else 'No source documents found.'}"

            st.chat_message("Dishy Bot").markdown(result_to_show)
            st.session_state.messages.append({'role': 'Dishy Bot', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error in processing the request: {e}") 

if __name__ == "__main__":
    main()
