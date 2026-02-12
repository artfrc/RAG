from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ollama import Client
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "docs" / "Documentacao.pdf"
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
EMBEDDINGS = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
COLLECTION_NAME = "urano_doc"

def load_and_split_doc():
   loader = PyPDFLoader(str(DATA_PATH))
   docs = loader.load()
   if docs:
    print(f"Loaded {len(docs)} pages.\n") # number of pages
    print("Content of the first page:")
    print(f"{docs[0].page_content}\n") # Print all content of first page
    print(f"Metadata of the first page: {docs[0].metadata}\n")
   else:
      print("Failed to load documents.")
      
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
   all_splits = text_splitter.split_documents(docs)
   return all_splits

def embedding_function(all_splits):
   vector_store = Chroma(embedding_function= EMBEDDINGS,
                         persist_directory = str(PERSIST_DIRECTORY),
                         collection_name = COLLECTION_NAME)
   _ = vector_store.add_documents(documents = all_splits)
   
   return vector_store
   
def process_user_input(vector_store):
         question = "Qual o CPX da doc?"
         retrieved_docs = vector_store.similarity_search(question, k=2)
         
         return retrieved_docs, question
         

def main():
   
   # loading and splting the text in chunks
   splits = load_and_split_doc()
   
   # creating vectors to store data using Chroma, AI-native open-source
   vector_store = embedding_function(splits)
     
   # to process user input
   retrieved_docs, question = process_user_input(vector_store)
   
   # generating response
   docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
   promptJson = {
      "question": question,
      "context": docs_content
   }
   prompt = json.dumps(promptJson, ensure_ascii = False)
   print(f">>> PROMPT, {prompt}")
   
   model = 'mistral'
   client = Client(
      host="http://localhost:11434"
      )
   response = client.chat(model=model, messages=[
               {
                  'role': 'user',
                  'content': prompt,
               },
         ])
   print(response.message.content)

if __name__ == "__main__":
   main()
   