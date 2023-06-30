import streamlit as st
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from streamlit_chat import message
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')


output_summary = ""
def on_file_upload():
    if st.session_state['file_uploader'] is not None:
        documents = [st.session_state['file_uploader'].read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(documents)
        chain = load_summarize_chain(llm, chain_type="map_reduce")

        output_summary = count_tokens_run(chain, docs)
        print(output_summary)


# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Form input and query
# openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not uploaded_file)
openai_api_key = "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"

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()

        # prompt_template = """
        # Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        #
        # {context}
        #
        # Question: {question}
        # Answer in Chinese:"""
        # PROMPT = PromptTemplate(
        #     template=prompt_template, input_variables=["context", "question"]
        # )
        # qa_chain = load_qa_chain(llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff", prompt=PROMPT)
        # qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=retriever)

        return qa.run(query_text)


def count_tokens_run(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'====Spent a total of {cb.total_tokens} tokens=====')
    return result


def on_input_change():
    user_input = st.session_state.user_input
    if user_input and openai_api_key.startswith('sk-'):
        st.session_state.past.append(user_input)
        # try:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, user_input)
            st.session_state.generated.append(response)
            st.session_state.user_input = ""
        # except Exception as e:
        #     st.error(e)


def on_clear_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


st.session_state.setdefault('past', [])
st.session_state.setdefault('generated', [])

chat_placeholder = st.empty()

with chat_placeholder.container():
    message('Hello, I am a bot. I am here to help you with your questions.', )
    if output_summary != "":
        message(output_summary)

    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i],
            key=f"{i}",
            allow_html=True
        )

    st.button("Clear message", on_click=on_clear_click)

st.text_input("Enter a short question:", placeholder='Please provide a short question.', on_change=on_input_change,
              key="user_input", disabled=not uploaded_file)
