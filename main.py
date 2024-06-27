from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def load_documents():
    path = ("D:\\tech\\RAG\\Dataset\\Signs_and_symptoms.pdf")
    Loader = PyPDFLoader(path)
    return Loader.load()


documents = load_documents()



def split_documents():
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size = 800,
       chunk_overlap = 80,
       length_function = len,
       is_separator_regex= False,
   )
   return text_splitter.split_documents(documents)
    
    
chunks = split_documents()
print(f"from {len(documents)} documents to {len(chunks)}")
print(chunks[0])


# def get_embeddings():
#     embedding = BedrockEmbeddings(credentials_profile_name="Default", region_name="us-east-1")
#     return embedding

# def build_db():
#     Chroma_path = ""
#     db = Chroma(persist_directory=Chroma_pa
# th,embedding_function=get_embeddings())
#     db.add_documents(new_chunks,ids=new_chunks_ids)
#     db.persist()

def get_embeddings():
            # Initialize the embedding model (you can use any model you prefer)
            embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

            # Convert chunks into embeddings
            chunk_texts = [chunk.page_content for chunk in chunks]
            chunk_embeddings = embedder.embed_documents(chunk_texts)
            return chunk_embeddings,chunk_texts
            
            
def store_embedding_chroma():
   get_embeddings()
   collection_name = "my_documents"
   chroma = Chroma(collection_name=collection_name)
   chroma.store_embeddings(chunk_embeddings, chunk_texts)







def prompt_template():
    template = f"""
    Answer me the question based on the context below, if you can't answer
    the question than simple write i don't Know
    
    here's the context : {context}
    
    Question:{question}
    """
 
 
    prompt = PromptTemplate.from_template(template)
    prompt.format(context=related_context ,question=query)