import os
from pinecone import Pinecone, Index, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI


from langchain_community.llms import OpenAI
from langchain.schema import Document
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain

def get_pdf_text(pdf_doc):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def create_docs(user_pdf_list, unique_id):
    """Create documents from uploaded PDFs."""
    docs = []
    for pdf_file in user_pdf_list:
        text = get_pdf_text(pdf_file)
        docs.append(Document(
            page_content=text,
            metadata={
                "name": pdf_file.name,
                "size": pdf_file.size,
                "unique_id": unique_id
            }
        ))
    print(docs)  # This should show Document objects, not dictionaries.
    return docs

def create_embeddings_load_data():
    """Create an instance of HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def initialize_pinecone(api_key, environment):
    """Return a Pinecone instance."""
    return Pinecone(api_key=api_key, environment=environment)

def push_to_pinecone(api_key, environment, index_name, embeddings, docs):
    """Push document embeddings to Pinecone."""
    pc = initialize_pinecone(api_key, environment)

    # Check if the index exists, create if necessary
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension must match embedding model
            spec=ServerlessSpec(cloud="aws", region=environment)
        )

    # Access the index
    index = pc.Index(index_name)

    # Prepare vectors for upsert
    vectors = [
    {
        "id": f"doc-{idx}",
        "values": embeddings.embed_query(doc.page_content),
        "metadata": {"page_content": doc.page_content, **doc.metadata}
    }
    for idx, doc in enumerate(docs)
    ]

    for idx, doc in enumerate(docs):
        print(f"Doc {idx}: {type(doc)} - {doc}")

    if isinstance(doc, dict):
        page_content = doc.get("page_content", "")
    else:
        page_content = doc.page_content

    print(type(doc))  # Should be <class 'langchain.schema.Document'>

    # Debugging log
    print(f"Upserting the following vectors:\n{vectors}")
    

    # Upsert vectors into the index
    index.upsert(vectors)
    print("Upsert completed.")

def similar_docs(query, k, api_key, environment, index_name, embeddings, unique_id):
    pc = initialize_pinecone(api_key, environment)
    index = pc.Index(index_name)

    # Generate query vector
    query_vector = embeddings.embed_query(query)

    # Fetch results
    results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True
    )

    documents = []
    for match in results.get("matches", []):
        metadata = match["metadata"]
        page_content = metadata.pop("page_content", "")  # Remove if exists
        documents.append((Document(page_content=page_content, metadata=metadata), match["score"]))

    return documents

    print(f"Query Results: {results}")
    return [(match["metadata"], match["score"]) for match in results.get("matches", [])]


def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run([current_doc])  # Pass current_doc directly.

