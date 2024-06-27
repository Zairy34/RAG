import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

class RAG:
    def __init__(self, path_to_data, path_to_db, query=None):
        self.path_to_data = path_to_data
        self.persist_directory = path_to_db
        self.query = query
        self.db = None
        self.documents = None
        self.chunks = None
        self.chunks_text = None
        self.chunks_embedding = None

    def model(self, input_text):
        llm = Ollama(model="llama2")
        response = llm.invoke(input_text)
        return response
    
    def load_documents(self):
        loader = PyPDFDirectoryLoader(self.path_to_data)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} documents")
    
    def text_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Split {len(self.documents)} documents into {len(self.chunks)} chunks")
    
    def get_embeddings(self):
        embedder = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.chunks_text = [chunk.page_content for chunk in self.chunks]
        self.chunks_embedding = embedder.embed_documents(self.chunks_text)
        print("Embeddings created successfully!")
        return embedder
    
    def embedding_to_db(self):
        embedder = self.get_embeddings()
        if not os.path.exists(self.persist_directory):
            os.mkdir(self.persist_directory)
        self.db = FAISS.from_documents(self.chunks, embedder)
        self.db.save_local(self.persist_directory)
        print("Saved successfully!")
    
    def load_db(self):
        embedder = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.load_local(self.persist_directory, embedder, allow_dangerous_deserialization=True)
        print("Loaded successfully")
    
    def search_documents(self):
        retriever = self.db.as_retriever()
        query_embedding = self.db.embed_query(self.query)
        docs = retriever.retrieve(query_embedding, top_k=5)
        for i, doc in enumerate(docs):
            print(f"Document {i + 1}:\n{doc.page_content}\n")
        return docs

    def run_query(self, query):
        self.query = query
        relevant_docs = self.search_documents()
        context = " ".join([doc.page_content for doc in relevant_docs])
        response = self.model(context + "\n\n" + self.query)
        print("Generated Response:\n", response)

if __name__ == "__main__":
    path_to_data = "C:\\Users\\zaidg\\Videos\\Project\\All data"    
    path_to_db = "D:\\DB"    
    obj = RAG(path_to_data=path_to_data, path_to_db=path_to_db)
    
    obj.load_db()

    while True:
        query = input("Please enter a query here: ")
        obj.run_query(query)
