import pinecone

# Step 1: Upsert Documents with Metadata
def upsert_documents_to_pinecone(documents, pinecone_index):
    """
    Upserts documents into the Pinecone index with metadata.
    :param documents: List of document dictionaries containing 'id', 'vector', 'content', and 'name'.
    :param pinecone_index: The Pinecone index instance.
    """
    for doc in documents:
        pinecone_index.upsert(
            items=[
                {
                    'id': doc['id'],
                    'values': doc['vector'],  # Vector embedding
                    'metadata': {
                        'page_content': doc['content'],  # Document content
                        'name': doc['name']  # File name or document name
                    }
                }
            ]
        )
    print("Documents upserted successfully.")

# Step 2: Query Pinecone and Parse Results
def query_pinecone_index(query_vector, pinecone_index, top_k=5):
    """
    Queries the Pinecone index and parses results for metadata and match scores.
    :param query_vector: The query vector for similarity search.
    :param pinecone_index: The Pinecone index instance.
    :param top_k: Number of top matches to retrieve.
    :return: Parsed query results.
    """
    # Query Pinecone index
    query_results = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # Parse results
    parsed_results = []
    if query_results and 'matches' in query_results:
        for match in query_results['matches']:
            # Extract metadata and score
            file_name = match['metadata'].get('name', 'Unknown File')
            page_content = match['metadata'].get('page_content', 'No Content Found')
            match_score = match['score']

            # Append to parsed results
            parsed_results.append({
                'file_name': file_name,
                'content_snippet': page_content[:200],  # Preview snippet
                'match_score': round(match_score, 2)
            })

            # Debugging output
            print(f"\ud83d\udd0d File Name: {file_name}")
            print(f"Match Score: {match_score:.2f}")
            print(f"Content Snippet: {page_content[:200]}...\n")
    else:
        print("No matches found.")

    return parsed_results

# Step 3: Format and Display Results in Browser
def display_results_in_browser(parsed_results):
    """
    Formats and displays parsed results in the browser.
    :param parsed_results: List of dictionaries containing file_name, content_snippet, and match_score.
    """
    print("Total Resumes Found:", len(parsed_results))
    for idx, result in enumerate(parsed_results):
        print(f"\n\ud83d\udd0d Resume {idx + 1}")
        print(f"File Name: {result['file_name']}")
        print(f"Match Score: {result['match_score']}")
        print(f"Content Snippet: {result['content_snippet']}\n")
