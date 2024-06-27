from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from sentence_transformers import util

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
print(chunks[0])

# Step 2: Convert Chunks into Embeddings 🧠✨
def get_embeddings(chunks):
    embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embedder.embed_documents(chunk_texts)
    return chunk_embeddings, chunk_texts

chunk_embeddings, chunk_texts = get_embeddings(chunks)

# Step 3: Store Embeddings in Chroma DB 🗄️
def store_embeddings_chroma(chunk_embeddings, chunk_texts):
    collection_name = "my_documents"
    chroma = Chroma(collection_name=collection_name)
    chroma.store_embeddings(chunk_embeddings, chunk_texts)

store_embeddings_chroma(chunk_embeddings, chunk_texts)

# Step 4: Set Up LLAMA2 Model 🦙
model_name = "your-llama2-model"  # Replace with the actual model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 5: Query Chroma DB and Generate a Response 🔍📝
def search_documents(query, chunk_embeddings, chunk_texts):
    embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = embedder.embed_documents([query])[0]
    scores = [util.cos_sim(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
    best_doc_idx = scores.index(max(scores))
    return chunk_texts[best_doc_idx]

def generate_response(query, context):
    template = """
    Answer the question based on the context below. If you can't answer, simply write "I don't know."
    
    Here's the context: {context}
    
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    prompt_text = prompt.format(context=context, question=query)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    output = model.generate(**inputs)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example query
query = "What are the signs and symptoms?"
related_context = search_documents(query, chunk_embeddings, chunk_texts)
response = generate_response(query, related_context)
print(response)
