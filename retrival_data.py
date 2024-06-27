from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.chroma import Chroma
from sentence_transformers import util
from langchain.prompts import PromptTemplate
import faiss
import numpy as np

def Load_documents():
    path = ("D:\\tech\\RAG\\Dataset\\Signs_and_symptoms.pdf")
    Loader = PyPDFLoader(path)
    return Loader.load()

documents = Load_documents()
print(len(documents))


def split_documents():
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size = 800,
       chunk_overlap = 80,
       length_function = len,
       is_separator_regex= False,
   )
   return text_splitter.split_documents(documents)
    


chunks = split_documents()
print(f"from {len(documents)} Document to {len(chunks)} Chunks ")


def get_embeddings(chunks):
   embedders = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
   chunks_text  = [chunk.page_content for chunk in chunks]
   chunks_embedding =  embedders.embed_documents(chunks_text)
   return chunks_text , chunks_embedding



chunks_text , chunks_embedding = get_embeddings(chunks)

query = "what actaually are signs and symptoms"




def store_embeddings_faiss(chunk_embeddings, chunk_texts):
    d = chunk_embeddings.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(np.array(chunk_embeddings))  # add vectors to the index
    return index, chunk_texts

index, chunk_texts = store_embeddings_faiss(chunks_embedding, chunks_text)


def search_documents(query, index, chunk_texts):
    embedder = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedder.embed_documents([query])[0]
    distances, indices = index.search(np.array([query_embedding]), k=3)
    best_doc_idx = indices[0][0]
    return chunk_texts[best_doc_idx]

context = search_documents(query,index,chunks_text)
print(context)

# def store_embeddings_chroma(chunks_text,chunks_embedding):
#     chroma_path = "D:\\tech\\RAG"
#     collection_name = "my documents"
#     chroma = Chroma(collection_name=collection_name,persist_directory=chroma_path)
#     chroma.store_embeddings(chunks_embedding,chunks_text)
#     print("Sucessfully saved")
    

# store_embeddings_chroma(chunks_text,chunks_embedding)
# def search_embeddings(querychunks_text,chunk_embeddings):
#        embedders = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
#        query_embedding = embedders.embed_documents([query])[0]
#        scores = [util.cos_sim(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
#        best_doc_idx = scores.index(max(scores))
#        return chunks_text[best_doc_idx]


# context  = search_embeddings(query=query,chunk_embeddings=chunks_embedding,chunks_text=chunks_text)
# print(context)


# def prompt_template(context,question):
#     template = """
#     Answer the question based on the context below. If you can't answer, simply write "I don't know."
    
#     Here's the context: {context}
    
#     Question: {question}
#     """
#     prompt = PromptTemplate.from_template(template)
#     prompt_text = prompt.format(context=context, question=query)

# input = prompt_template(context=context,question=query)


# #using ollama for running llama2
# model.predict(**input)