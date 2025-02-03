# import the necessary packages
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#================= step 1: upload the data =================
data_path = 'Data/'
def pdf_file_upload(data):
    loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = pdf_file_upload(data=data_path)
# print("length of pdf", len(documents))

#================= step 2: create chunk of data =================
def create_chunk(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap = 50)
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk
text_chunk = create_chunk(extracted_data=documents)
# print("lenght of chunk", len(text_chunk))

#=================step 3: create embedding of that chunks=================

def create_embedding():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding

embedding_model = create_embedding()

#step 4: store those embedding into database
DB_PATH = 'VectorStore/data_base'
db = FAISS.from_documents(text_chunk, embedding_model)
db.save_local(DB_PATH)