import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#=================step 4: setup LLm (mistral with huggingface)=================
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":512}
    )
    return llm

# #=================step 4: Connect LLM with FAISS and create chain=================
custom_prompt_templates = """
You are an AI assistant that provides step-by-step cooking instructions based on the given context.  
Use the retrieved context below to answer the user's question.  

Context: {context}  
Question: {question}  

Instructions:  
- First, list all required ingredients** clearly before giving the steps.  
- Then, provide a step-by-step cooking method** in a structured format.  
- If the context does not contain enough information respond with: I don't have enough information to answer this."  
- Do not make up any ingredients or steps**—only use retrieved information.   
"""

def set_custom_prompt(custom_prompt_templates):
    prompt = PromptTemplate(template=custom_prompt_templates, input_variables=["context", "question"])
    return prompt

# load the database
DB_custom_path = 'VectorStore/data_base'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_custom_path, embedding_model,allow_dangerous_deserialization=True)
 
# create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs = {'prompt':set_custom_prompt(custom_prompt_templates)})

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
