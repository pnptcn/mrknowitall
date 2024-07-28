import os
import textwrap

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core import StorageContext

documents = SimpleDirectoryReader("~/data/south pole", recursive=True).load_data()

dataset_path = "~/store"

# Create an index over the documents
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()

question = "What are the main unanswered questions?"
prompt = "Provide a list of highly specific Google Dorks to research the main unanswered questions below:"

while True:
    response = query_engine.query(question)
    research = query_engine.query(prompt + "\n\n" + response)
    print(research)
    exit()
