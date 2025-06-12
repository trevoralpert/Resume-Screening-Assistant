import streamlit as st
import uuid
from utils4 import *
from dotenv import load_dotenv
import os

load_dotenv()

# Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():
    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance ✋")
    st.subheader("I can help you in the resume screening process")

    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...", key="1")
    document_count = st.text_input("No. of 'RESUMES' to return", key="2")
    pdf = st.file_uploader("Upload resumes here (PDF only):", type=["pdf"], accept_multiple_files=True)

    if st.button("Help me with the analysis"):
        with st.spinner("Wait for it..."):
            try:
                st.session_state['unique_id'] = uuid.uuid4().hex
                unique_id = st.session_state['unique_id']

                docs = create_docs(pdf, unique_id)
                st.write(f"Total Resumes Uploaded: {len(docs)}")

                embeddings = create_embeddings_load_data()

                # Get Pinecone credentials
                pinecone_apikey = os.getenv("PINECONE_API_KEY")
                pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
                pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

                # Push data to Pinecone
                push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs)

                # Fetch relevant documents from Pinecone
                relevant_docs = similar_docs(
                    query=job_description,
                    k=int(document_count),
                    api_key=pinecone_apikey,
                    environment=pinecone_environment,
                    index_name=pinecone_index_name,
                    embeddings=embeddings,
                    unique_id=unique_id
                )

                if not relevant_docs:
                    st.warning("No relevant resumes were found. Please try a different job description or upload additional resumes.")


                for idx, (doc, score) in enumerate(relevant_docs):
                    st.subheader(f"👉 Resume {idx + 1}")
                    st.write(f"**File Name:** {doc.metadata['name']}")
                    st.info(f"**Match Score:** {score:.2f}")

                with st.expander("View Summary"):
                    current_doc = doc
                    summary = get_summary(current_doc)
                    st.write(f"**Summary:** {summary}")


                st.success("Analysis complete! Hope I saved you some time.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
