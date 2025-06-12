from flask import Flask, request, jsonify
from utilstest import upsert_documents_to_pinecone, query_pinecone_index, display_results_in_browser
import pinecone

app = Flask(__name__)

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="your-environment")
pinecone_index = pinecone.Index("your-index-name")

@app.route("/upsert", methods=["POST"])
def upsert_documents():
    """
    Endpoint to upsert documents into the Pinecone index.
    """
    data = request.get_json()
    documents = data.get("documents", [])

    if not documents:
        return jsonify({"error": "No documents provided."}), 400

    upsert_documents_to_pinecone(documents, pinecone_index)
    return jsonify({"message": "Documents upserted successfully."}), 200

@app.route("/query", methods=["POST"])
def query_documents():
    """
    Endpoint to query documents from the Pinecone index.
    """
    data = request.get_json()
    query_vector = data.get("query_vector", [])
    top_k = data.get("top_k", 5)

    if not query_vector:
        return jsonify({"error": "Query vector is required."}), 400

    parsed_results = query_pinecone_index(query_vector, pinecone_index, top_k=top_k)
    return jsonify({"results": parsed_results}), 200

if __name__ == "__main__":
    app.run(debug=True)
