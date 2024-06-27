from langchain_community.llms import Ollama
llm = Ollama(model = "llama2")
response = llm.invoke("hello!")
print(response)