from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
import faiss
import numpy as np
from langchain_community.llms import ollama


# Step 1: Load and Split Documents 📚✂️
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


# Step 2: Convert Chunks into Embeddings 🧠✨
def get_embeddings(chunks):
    embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embedder.embed_documents(chunk_texts)
    return chunk_embeddings, chunk_texts

chunk_embeddings, chunk_texts = get_embeddings(chunks)

# Step 3: Store Embeddings in FAISS 🗄️
def store_embeddings_faiss(chunk_embeddings, chunk_texts):
    chunk_embeddings_np = np.array(chunk_embeddings)  # Convert list of embeddings to NumPy array
    d = chunk_embeddings_np.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(chunk_embeddings_np)  # add vectors to the index
    return index, chunk_texts

index, chunk_texts = store_embeddings_faiss(chunk_embeddings, chunk_texts)

# Step 5: Query FAISS and Generate a Response 🔍📝
def search_documents(query, index, chunk_texts):
    embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = embedder.embed_documents([query])[0]
    distances, indices = index.search(np.array([query_embedding]), k=3)
    best_doc_idx = indices[0][0]
    return chunk_texts[best_doc_idx]



query = "What are the signs and symptoms?"
related_context = search_documents(query, index, chunk_texts )

def generate_response(query, context):
    template = """
    Answer the question based on the context below. If you can't answer, simply write "I don't know."
    
    Here's the context: {context}
    
    Question: {question}
    """
    
    prompt = PromptTemplate.from_template(template)
    prompt_text = prompt.format(context=context, question=query)
    return prompt_text
    
input = generate_response(query=query,context=related_context)
print(input)
llm = ollama(model = "llama2")
print("========================================================================================================")
results = llm.invoke(input)
print(results)

    # inputs = tokenizer(prompt_text, return_tensors="pt")
    # output = model.generate(**inputs)
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    # return response

# Example query




