import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEmebeddings
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub


#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,
                      "id":filename.id,
                      "type=":filename.type,
                      "size":filename.size,
                      "unique_id":unique_id},
        ))

    return docs

#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = openAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

#Function to push data to Vector Store - Pinecone here
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):
    # Initialize the Pinecone client
    """Push document embeddings to Pinecone."""
    pinecone_client = pinecone.Client(api_key=pinecone_apikey, environment=pinecone_environment)
    index = pinecone_client.index(pinecone_index_name)

    for idx, doc in enumerate(docs):
        vector = {
            "id": f"doc-{idx}",
            "values": embeddings.embed_query(doc.page_content),
            "metadata": doc.metadata
        }
        index.upsert(vectors=[vector])
    print("Upsert completed.")

def pull_from_pinecone(pinecone_apikey, pinecone_index_name, host):
    pc = Pinecone(api_key=pinecone_apikey)
    print(f"Connecting to index '{pinecone_index_name}' with host '{host}'...")
    index = pc.Index(name=pinecone_index_name, host=host)
    return index


def similar_docs(query,k,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,unique_id):
    """Fetch similar documents from Pinecone based on query."""
    pc = Pinecone(api_key=pinecone_apikey, environment=pinecone_environment)
    index = pc.index(pinecone_index_name)

    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True,
        filter={"unique_id": unique_id}
    )
    return [(match.metadata, match.score) for match in results["matches"]]


#Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary