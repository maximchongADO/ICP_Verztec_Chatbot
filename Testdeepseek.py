import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader , PyPDFLoader

load_dotenv()
api_key="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"
model="deepseek-r1-distill-llama-70b"
deepseek = ChatGroq(api_key=api_key,model_name=model)

print(deepseek.invoke("hello"))

parser = StrOutputParser()

deepseek_chain = deepseek | parser

print(deepseek_chain.invoke("hello"))

