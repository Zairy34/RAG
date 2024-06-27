from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import os



class RAG:
    def __init__(self,path_to_data,path_to_db,query):
        self.path_to_data = path_to_data
        self.presist_directory = path_to_db
        self.query = query
        
            
    def load_documents(self):
        Loader = PyPDFDirectoryLoader(self.path_to_data)
        self.documents = Loader.load()
        print(len(self.documents))
        
    def text_spliter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        add_start_index=True,
        strip_whitespace=True,
            )
        self.chunks =  self.text_splitter.split_documents(self.documents)
        
        print(f"from {len(self.documents)} to {len(self.chunks)} chunks")
        
    def get_embeddings(self):
        embedder = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
        self.chunks_text = [chunk.page_content for chunk in self.chunks]
        self.chunks_embedding = embedder.embed_documents(self.chunks_text)
        print("embedding created sucessfully!😮")
        return embedder

    def embedding_to_db(self):
        embedder = self.get_embeddings()
        if not os.path.exists(self.presist_directory):
            os.mkdir(self.presist_directory)
        self.db = FAISS.from_documents(self.chunks,embedder)
        self.db.save_local(self.presist_directory)
        print("database Saved sucessfully! 🤩")
        
        
    def load_db(self):
        self.db = FAISS.load_local(self.presist_directory, HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
        print("Loaded sucessfully")
        
    def search_documents(self):
        # retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        # retrieved_docs = retriever.invoke(self.query)
        # print(retrieved_docs[0].page_content)

        query_embedding = self.db._embed_query(self.query)
        self.docs = self.db.similarity_search_by_vector(query_embedding,k=3)
        for i, doc in enumerate(self.docs):
            print(f"Document {i + 1}:\n{doc.page_content}\n")
        print(self.docs[0].page_content)
        
        
    # def Prompt_Template(self):
    #     context = "\n".join(
    #         [f"Document {i + 1}:\n{doc.page_content}\n" for i, doc in enumerate(self.docs)]
    #     )
    #     template = f"""
    #         Answer the question based on the context below. If you can't answer, simply write "I don't know."
    
    #         Here's the context: {context}
    
    #         Question: {self.query}
    #     """
    #     print(template)
    #     return template
    
    # def Model(self):
    #     template = self.Prompt_Template()
    #     print("inside model call function🤩")
    #     #print(template)
    #     llm = Ollama(model = "llama2")
    #     response = llm.invoke(template)
    #     print(response)
        
        
        # retriver = self.db.as_retriever()
        # query_embedding = self.db._embed_query(self.query)
        # self.docs = retriver.retrive(query_embedding, top_k=5)
        # for i, doc in enumerate(self.docs):
        #     print(f"Document {i + 1}:\n{doc.page_content}\n")
        # return self.docs


if __name__ == "__main__":
        path_to_data = "C:\\Users\\zaidg\\Videos\\Project\\All data"    
        path_to_db = "D:\\tech\\model Deployment\\SAAS Development\\RAG DATA\\6"    
        obj = RAG(path_to_data=path_to_data,path_to_db=path_to_db,query=None)
        obj.load_documents()
        obj.text_spliter()
        obj.embedding_to_db()
        obj.load_db()
        while True:
            query = input("please enter a query here  :    ")
            obj.query = query
            obj.search_documents()      
            # obj.Prompt_Template()
   #         obj.Model()
            
            
        
        
        
        
        
   