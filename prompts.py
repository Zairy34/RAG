from langchain_community.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
from sentence_transformers import util


        
        

query = "Tell me the types of acne"        


def load_documents():
    url = "https://www.niams.nih.gov/health-topics/all-diseases"
    Loader = WebBaseLoader(url)
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
print(f"from {len(documents)} to {len(chunks)} chunks")

def get_embeddings():
    embedders = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    chunks_text = [chunk.page_content for chunk in chunks]
    chunks_embedding = embedders.embed_documents(chunks_text)
    return chunks_text,chunks_embedding

chunks_text,chunks_embedding = get_embeddings()
print("done")

def embeddings_to_database(chunk_embeddings, chunk_texts, index_file='faiss_index.index'):
    d = chunk_embeddings.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(np.array(chunk_embeddings))  # add vectors to the index
    faiss.write_index(index, index_file)  # save the index locally
    return index, chunk_texts

# Save embeddings locally
index_file = 'faiss_index.index'
index, chunk_texts = embeddings_to_database(np.array(chunks_embedding), chunks_text, index_file)
print(f"FAISS index saved locally as {index_file} 📂")


def search_documents(query, index, chunk_texts):
    embedder = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedder.embed_documents([query])[0]
    distances, indices = index.search(np.array([query_embedding]), k=3,)
    best_doc_idx = indices[0][0]
    return chunk_texts[best_doc_idx]

context = search_documents(query,index,chunks_text)
print(context)

def Template(context,question):
    template = """
    Answer the question based on the context below. If you can't answer, simply write "I don't know."
    
    Here's the context: {context}
    
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    prompt_text = prompt.format(context=context, question=query)
    print(prompt_text)
    return prompt_text
    
input = Template(context=context,question=query)

def model(input):
        llm = Ollama(model= "llama2")
        print(llm.invoke(input))
        
model(input=input)