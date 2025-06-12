import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    return docs

def create_embeddings_load_data():
    """Create an instance of HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def initialize_pinecone(api_key, environment):
    """Initialize the Pinecone client."""
    return Pinecone(api_key=api_key)

def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    """Push document embeddings to Pinecone."""
    # Initialize Pinecone
    pc = initialize_pinecone(pinecone_apikey, pinecone_environment)

    # Check if the index exists
    if pinecone_index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=pinecone_index_name,
            dimension=384,  # Ensure the dimension matches your embedding model
            spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
        )

    index = pc.index(pinecone_index_name)

    # Prepare vectors for upsert
    vectors = [
        {
            "id": f"doc-{idx}",
            "values": embeddings.embed_query(doc.page_content),
            "metadata": doc.metadata
        } for idx, doc in enumerate(docs)
    ]

    # Upsert vectors into the index
    index.upsert(vectors)
    print("Upsert completed.")

def similar_docs(query, k, pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, unique_id):
    """Fetch similar documents from Pinecone based on query."""
    # Initialize Pinecone
    pc = initialize_pinecone(pinecone_apikey, pinecone_environment)
    index = pc.index(pinecone_index_name)

    # Query the index
    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True,
        filter={"unique_id": unique_id}
    )
    return [(match["metadata"], match["score"]) for match in results["matches"]]

def get_summary(current_doc):
    """Generate a summary for a given document."""
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run([current_doc])
