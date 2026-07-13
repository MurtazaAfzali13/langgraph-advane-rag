import os

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader

from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls=[
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
]

docs=[WebBaseLoader(url).load()  for url in urls]

docs_list=[item for sublist in docs for item in sublist]

text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200,
    chunk_overlap=0
)

doc_splits=text_splitter.split_documents(docs_list)

embeddings=OpenAIEmbeddings(
      model="openai/text-embedding-3-small",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)


# vectorestore=Chroma.from_documents(
#     documents=doc_splits,
#     embedding=embeddings,
#     collection_name="rag-chroma",
#     persist_directory="./.chroma"
# )

retriver=Chroma(
    collection_name="rag-chroma",
    persist_directory="./chroma",
     embedding_function=embeddings,
).as_retriever()

