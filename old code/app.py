import streamlit as st
import uuid
from utils import *
from dotenv import load_dotenv
import os

load_dotenv()

#Creating session variabels
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():

    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance...✋ ")
    st.subheader("I can help you in resume screening process")

    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...",key="1")
    document_count = st.text_input("No. of 'RESUMES' to return",key="2")
    # Upload the Resumes (pdf files)
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed",type=["pdf"],accept_multiple_files=True)

    submit=st.button("Help me with the analysis")

    if submit:
        with st.spinner("Wait for it..."):
            
            st.write("our process")
            #Creating a unique ID, so that we can use to query and get  only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex
            st.write(st.session_state['unique_id'])
            unique_id = st.session_state['unique_id']

            #Create a documents list out of all the user uploaded pdf files
            docs=create_docs(pdf,st.session_state['unique_id'])
            st.write(docs)

            #Displauing the count of resumes that have been uploaded
            st.write(len(docs))


            #Create embeddings instance
            embeddings=create_embeddings_load_data()
                
            pinecone_apikey = os.getenv("PINECONE_API_KEY")
            pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
            pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")      
            
            #Push data to PINECONE
            push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs)

            #Fetch relevant documents from PINECONE
            relevant_docs = similar_docs(query=job_description,
                    k=int(document_count),
                    pinecone_apikey=pinecone_apikey,
                    pinecone_environment=pinecone_environment,
                    pinecone_index_name=pinecone_index_name,
                    embeddings=embeddings,
                    unique_id=unique_id)

            st.write(relevant_docs)

            #Introducing a line separator
            st.write(":heavy_minus_sign:"*30)

            #For each item in relevant docs - we are displaying some info of it on the UI
            for item in range(len(relevant_docs)):

                st.subheader("👉 "+str(item+1))

                #Displaying Filepath
                st.write("**File** : "+relevant_docs[item][0].metadata['name'])

                #Introducing Expander feature
                with st.expander('Show me 👀'):
                    st.info("**Match Score** : "+str(relevant_docs[item][1]))
                    #st.write("***"+relevant_docs[item][0].page_content)

                    # Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain
                    summary = get_summary(relevant_docs[item][0])
                    st.write("**Summary** : "+summary)

        st.success("Hope I was able to save your time")



#Invoking main function
if __name__ == '__main__':
    main()
