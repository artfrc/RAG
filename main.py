from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ollama import Client
import json

DATA_PATH = "/home/costaarthur/me/projetos/urano/RAG/my-rag/docs/Document 5.pdf"
PERSIST_DIRECTORY = "/home/costaarthur/me/projetos/urano/RAG/my-rag/chroma_db"
EMBEDDINGS = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
COLLECTION_NAME = "urano_doc"

def load_and_split_doc():
   loader = PyPDFLoader(DATA_PATH)
   docs = loader.load()
   if docs:
    print(f"Loaded {len(docs)} pages.") # number of pages
    print("Content of the first page:")
    print(docs[0].page_content) # Print all content of first page
    print(f"Metadata of the first page: {docs[0].metadata}")
   else:
      print("Failed to load documents.")
      
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
   all_splits = text_splitter.split_documents(docs)
   return all_splits

def embedding_function(all_splits):
   vector_store = Chroma(embedding_function= EMBEDDINGS,
                         persist_directory = PERSIST_DIRECTORY,
                         collection_name = COLLECTION_NAME)
   _ = vector_store.add_documents(documents = all_splits)
   
   return vector_store
   
def process_user_input(vector_store):
         question = "My question"
         retrieved_docs = vector_store.similarity_search(question, k=2)
         
         return retrieved_docs
         

def main():
   
   # loading and splting the text in chunka
   splits = load_and_split_doc()
   
   # creating vectors to store data using Chroma,  AI-native open-source
   vector_store = embedding_function(splits)
     
   # to process user input
   retrieved_docs = process_user_input(vector_store)
   
   # generating response
   docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
   promptJson = {
      "question": ""
   }
   
   
   
   
   
if __name__ == "__main__":
   main()
   