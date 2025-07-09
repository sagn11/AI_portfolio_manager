import streamlit as st
from Portfolio_developer import portfolio_developer
from Sentiment_analyzer import sentiment_anal
from RAG_doc import RAG

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Load model
llm = Ollama(model="llama3.2")
parser = StrOutputParser()

# Prompt template to classify task type
router_prompt = ChatPromptTemplate.from_template("""
You are a function-routing assistant. The user will type a finance-related query. Your job is to classify it as one of:

- "portfolio" → for investment planning, goals, growth, stocks, risk, asset allocation
- "sentiment" → for news headlines, stock analysis, positive/negative tone, media events
- "rag" → for questions about documents, reports, balance sheets, financial statements

Respond with ONLY ONE WORD: "portfolio", "sentiment", or "rag"
Do not add any explanation.

Query:
{query}
""")

st.set_page_config(page_title="AI Portfolio Manager", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stTextInput>div>div>input {
        text-align: center;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 AI Portfolio Manager")
st.markdown("Type your query below. This assistant routes your input to the correct financial engine.")

query = st.text_input("🧠 What would you like to do?", "")

if query:
    with st.spinner("Analyzing your query..."):
        route = (router_prompt | llm | parser).invoke({"query": query}).strip().lower()

    st.success(f"Routed to: `{route}`")

    if route == "portfolio":
        st.subheader("📊 Portfolio Recommendation:")
        output = portfolio_developer(query)
        st.write(output)

    elif route == "sentiment":
        st.subheader("📰 News Sentiment Analysis:")
        output = sentiment_anal(query)
        st.write(output)

    elif route == "rag":
        st.subheader("📄 Document-Based QA:")
        output = RAG(query)
        st.write(output)

    else:
        st.error("❌ Could not classify query. Try rephrasing.")
