import streamlit as st
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool
from decouple import config

from util import count_tokens_run

st.markdown("# Ask question and get answer from web🎉")

def run_agent(question):
    search = SerpAPIWrapper(serpapi_api_key=config('SERPAPI_KEY'))
    llm = OpenAI(temperature=0)
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description='google search'
        )
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
    return count_tokens_run(agent, question)


# Accept user questions/query
query = st.text_input("Ask questions and get answer from web :")
if query:
    st.markdown("## Question")
    st.write(query)
    st.markdown("## Answer")
    with st.spinner('Answering...'):
        st.write(run_agent(query))
