import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from decouple import config
from langchain.llms import OpenAI

import os

os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
llm = OpenAI(temperature=0)

# Page title
st.set_page_config(page_title='🦜🔗 Doc Summary')
st.title('🦜🔗 Get the Doc Summary')

st.session_state.setdefault('output_summary', "")
st.session_state.setdefault('token_count', "")


def on_file_upload():
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        docs = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings()
        # Create a vectorstore from documents
        db = Chroma.from_documents(docs, embeddings)
        # Create retriever
        st.session_state.retriever = db.as_retriever()

        chain = load_summarize_chain(llm, chain_type="map_reduce")

        st.session_state.output_summary = count_tokens_run(chain, docs)
    else:
        return None


def count_tokens_run(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.session_state.token_count = f'====Spent a total of {cb.total_tokens} tokens====='
    return result


# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
on_file_upload()

if st.session_state.output_summary != "":
    '''## Summary:'''
    st.write(st.session_state.output_summary)

if st.session_state.token_count:
    st.info(st.session_state.token_count)
