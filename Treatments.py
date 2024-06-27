from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
import os
#from langchain.docstore import InMemoryDocstore
from langchain.docstore import InMemoryDocstore

class RAG:
    def __init__(self,query,path_to_data,path_to_db):
        self.query = query
        self.path = path_to_data
        self.persist_directory = path_to_db
        
    def Model(self):
        llm = Ollama(model = "llama2")
        respone = llm.invoke(self.query)
        print(respone)
        
    def load_documents(self):
        Loader = PyPDFDirectoryLoader(self.path)
        self.documents =  Loader.load()
        
        
    def text_splitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex= False,
            )
        self.chunks =  self.text_splitter.split_documents(self.documents)
        
        print(f"from {len(self.documents)} to {len(self.chunks)} chunks")
        
    def get_embeddings(self):
        embedder = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
        self.chunks_text = [chunk.page_content for chunk in self.chunks]
        self.chunks_embedding = embedder.embed_documents(self.chunks_text)
        print(self.chunks_embedding[0])
        return embedder
    
    
    def embeddings_to_db(self):
        embedder = self.get_embeddings()
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        self.db = FAISS.from_documents(self.chunks, embedder)      
        self.db.save_local(self.persist_directory)
        print("completed sucessfully!")
        
    def load_db(self):
        self.db = FAISS.load_local(self.persist_directory, HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
        print("Loaded sucessfully")
        
        
    def retrieve_documents(self):
        # self.docs =self.db.similarity_search(self.query)
        retriever = self.db.as_retriever()
        self.docs = retriever.invoke(query)
        print(self.docs[0].page_content)

    
    
    



path = "C:\\Users\\zaidg\\Videos\\Project\\All data"
path_to_db = "D:\\tech\\RAG\\Dataset"  
obj = RAG(query=None,path_to_data=path,path_to_db=path_to_db)
obj.load_db()
while True:
    query = input("Please enter your query: ")
    obj.query = query
    obj.retrieve_documents()
# obj.load_documents()
# obj.text_splitter()
# obj.get_embeddings()
# obj.embeddings_to_db()


