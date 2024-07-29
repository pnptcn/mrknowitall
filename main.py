import os
import shutil
import time
from typing import Dict, List, Union

import backoff
import httpx
import nest_asyncio
import textract
from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore import VectorStore
from minio import Minio
from neo4j import GraphDatabase
from openai import OpenAI

nest_asyncio.apply()

client = Minio(
    "host.docker.internal:9000",
    access_key="miniouser",
    secret_key="miniosecret",
    secure=False,
)

if client.bucket_exists("datalake"):
    print("bucket exists")
else:
    print("make bucket")
    client.make_bucket("datalake")


# Retry decorator for handling timeouts
@backoff.on_exception(
    backoff.expo, (httpx.ReadTimeout, httpx.ConnectTimeout), max_tries=5
)
def retry_request(func, *args, **kwargs):
    return func(*args, **kwargs)


client = OpenAI(
    base_url="http://host.docker.internal:1234/v1", api_key="lm-studio", timeout=60.0
)

# DeepLake setup
vector_store = VectorStore(
    "s3://datalake",
    creds={
        "aws_access_key_id": "miniouser",
        "aws_secret_access_key": "miniosecret",
        "endpoint_url": "http://minio:9000",
    },
)

# Neo4j setup
neo4j_uri = "bolt://neo4j:7687"
neo4j_user = "neo4j"
neo4j_password = "securepassword"
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


def embedding_function(texts, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    if isinstance(texts, str):
        texts = [texts]

    texts = [t.replace("\n", " ") for t in texts]

    return [
        data.embedding
        for data in client.embeddings.create(input=texts, model=model).data
    ]


def add_to_vector_store(chunked_text, metadata, vector_store):
    vector_store.add(
        text=chunked_text,
        embedding_function=embedding_function,
        embedding_data=chunked_text,
        metadata=metadata,
    )


def initialize_vector_store(
    data_path: str = "/app/data", processed_path: str = "/app/processed_data"
):
    CHUNK_SIZE = 1000

    # Ensure the processed_path directory exists
    os.makedirs(processed_path, exist_ok=True)

    chunked_text = []
    metadata = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for file in filenames:
            try:
                full_path = os.path.join(dirpath, file)
                file_extension = os.path.splitext(file)[1].lower()

                if file_extension in [
                    ".txt",
                    ".doc",
                    ".docx",
                    ".pdf",
                    ".csv",
                    ".xls",
                    ".xlsx",
                ]:
                    # Use textract to extract text from the file
                    text = textract.process(full_path).decode("utf-8")

                    # Chunk the text
                    new_chunked_text = [
                        text[i : i + CHUNK_SIZE]
                        for i in range(0, len(text), CHUNK_SIZE)
                    ]
                    chunked_text += new_chunked_text
                    metadata += [
                        {"filepath": full_path, "filetype": file_extension}
                        for _ in range(len(new_chunked_text))
                    ]

                    # Move the processed file
                    new_path = os.path.join(
                        processed_path, os.path.relpath(full_path, data_path)
                    )
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    shutil.move(full_path, new_path)
                    print(f"Moved processed file to: {new_path}")
                else:
                    print(f"Skipping unsupported file type: {full_path}")
            except Exception as e:
                print(f"Error processing file {full_path}: {e}")

    if chunked_text:
        add_to_vector_store(chunked_text, metadata, vector_store)
        print(
            f"Added {len(chunked_text)} chunks from {len(metadata)} files to the vector store."
        )
    else:
        print("No valid documents found to add to the vector store.")


@backoff.on_exception(
    backoff.expo, (httpx.ReadTimeout, httpx.ConnectTimeout), max_tries=5
)
def query_vector_store(prompt: str) -> Union[Dict, Dataset]:
    results = vector_store.search(
        embedding_data=prompt, embedding_function=embedding_function
    )
    return results[0].text if results else ""


@backoff.on_exception(
    backoff.expo, (httpx.ReadTimeout, httpx.ConnectTimeout), max_tries=5
)
def generate_search_queries(context: str) -> List[str]:
    prompt = f"Based on the following context, generate 3 specific Google Dorks to research further:\n\n{context}\n\nGoogle Dorks:"

    stream = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        messages=[
            {
                "role": "system",
                "content": "Always output only the search queries, no other text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        stream=True
    )

    out = []
    for part in stream:
        content = part.choices[0].delta.content
        if content is not None:
            out.append(content)
            print(content, end="")

    return out


def analyze_graph_data(tx, query: str) -> List[Dict]:
    result = tx.run(query)
    return [dict(record) for record in result]


def disambiguate_data(data: List[Dict]) -> List[Dict]:
    # Implement disambiguation logic here
    return data


def resolve_edge_redundancies(tx):
    query = """
    MATCH (a)-[r1]->(b)<-[r2]-(a)
    WHERE type(r1) = type(r2)
    WITH a, b, collect(r1) + collect(r2) AS rels
    WHERE size(rels) > 1
    FOREACH (r IN tail(rels) | DELETE r)
    """
    tx.run(query)


def identify_enrichment_opportunities(data: List[Dict]) -> List[str]:
    # Implement logic to identify nodes/edges that need enrichment
    return ["Entity1", "Entity2", "Relationship1"]


def enrich_graph_data(tx, entity: str, new_data: Dict):
    query = f"""
    MERGE (n:{entity})
    SET n += $props
    """
    tx.run(query, props=new_data)


def main_research_loop():
    initialize_vector_store("/app/data", "/app/processed_data")

    while True:
        try:
            # 1. Analyze available data and formulate questions
            question = "What are the main unanswered questions in our current research?"
            context = query_vector_store(question)

            # 2. Generate search queries
            search_queries = generate_search_queries(context)
            print("Generated search queries:", search_queries)

            # 3. Perform web search and ingest new data (not implemented in this example)
            # new_data = perform_web_search(search_queries)
            # ingest_new_data(new_data)

            # 4. Analyze graph data
            with neo4j_driver.session() as session:
                graph_data = session.read_transaction(
                    analyze_graph_data, "MATCH (n) RETURN n"
                )

                # 5. Disambiguate data
                disambiguated_data = disambiguate_data(graph_data)

                # 6. Resolve edge redundancies
                session.write_transaction(resolve_edge_redundancies)

                # 7. Identify enrichment opportunities
                enrichment_targets = identify_enrichment_opportunities(
                    disambiguated_data
                )

                # 8. Enrich graph data
                for target in enrichment_targets:
                    enrichment_query = (
                        f"How can we enrich our knowledge about {target}?"
                    )
                    enrichment_info = query_vector_store(enrichment_query)
                    session.write_transaction(
                        enrich_graph_data, target, {"info": str(enrichment_info)}
                    )

            # Optional: Add a break condition or user input to stop the loop
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Retrying in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    main_research_loop()
