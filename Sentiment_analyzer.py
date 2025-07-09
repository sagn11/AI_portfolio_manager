import feedparser
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yfinance as yf
import pandas as pd
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
model=Ollama(model="llama3.2")
from langchain.schema.output_parser import StrOutputParser

def get_google_news_rss(company, num_articles=5):
    query = company.replace(" ", "+")
    feed_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(feed_url)
    
    return [entry.title for entry in feed.entries[:num_articles]]
sentiment_system_prompt = """
You are a financial sentiment analyst.

Your job is to analyze news headlines about a company and return one word only along with the company name:
- "good" if the overall sentiment is positive for the company
- "bad" if the headlines indicate issues, scandals, losses, etc.
- "neutral" if the news is mixed or non-impactful

Only return one of: "good", "bad", or "neutral"
"""
import pandas as pd
import time
import random
def sentiment_anal(portfolio):
    df = pd.read_csv(portfolio)
    tickers = df['ticker'].tolist() 

    sentiments = {}

    for ticker in tickers:
        headlines = get_google_news_rss(ticker)  # fetch for this ticker only
        if not headlines:
            print(f"⚠️ No headlines found for {ticker}. Skipping.")
            continue

        headline_string = "\n".join(headlines)
    
        prompt3 = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(sentiment_system_prompt),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        chain = prompt3 | model | StrOutputParser()
        sentiment = chain.invoke({"input": f"Analyze the sentiment of the following headlines:\n\n{headline_string}"})
    
        sentiments[ticker] = sentiment.strip().lower()
        time.sleep(random.uniform(3, 6))  # polite scraping

    # Flag stocks for review
    print("\n===  Sentiment Analysis Summary ===\n")
    for ticker in tickers:
        sentiment = sentiments.get(ticker, "unknown")
        print(f"{ticker}: {sentiment.capitalize()}")

        if sentiment in ["bad", "negative", "bearish"]:
            return "⚠️ Action: Review this company\n"
        elif sentiment == "unknown":
            return "❓ Action: Sentiment not available\n"
        else:
            return "✅ All good\n"
