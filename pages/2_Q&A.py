import streamlit as st
from streamlit_chat import message
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

llm = OpenAI(temperature=0)

st.session_state.setdefault('past', [])
st.session_state.setdefault('generated', [])


def generate_response(query_text):
    # Load document if file is uploaded
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
    if st.session_state.retriever:
        with st.spinner('Answering...'):
            # Create QA chain
            qa = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=st.session_state.retriever)
            qa_result = qa.run(query_text)
            return qa_result


def on_clear_msg_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


def on_input_change():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.past.append(user_input)
        response = generate_response(user_input)
        st.session_state.generated.append(response)
        st.session_state.user_input = ""


chat_placeholder = st.empty()
with chat_placeholder.container():
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i],
            key=f"{i}",
            allow_html=True
        )

    st.button("Clear message", on_click=on_clear_msg_click)

st.text_input("Enter a short question:", placeholder='Please provide a short question.',
              on_change=on_input_change,
              key="user_input")
