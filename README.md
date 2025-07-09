# AI_portfolio_manager
# AI Portfolio Manager 🧠📊

An **agentic AI tool** that helps users build a personalized stock portfolio, analyze associated sentiment, and explore company fundamentals—all powered by live web scraping, language models, and a RAG pipeline.

This project simulates the workflow of a retail quant analyst using open-access data and transformer-based models, with no paid APIs involved.

---

## 🧩 System Overview

The AI Portfolio Manager is structured around **three independent pipelines**:

---

### 1️⃣ Portfolio Developer 📈

Helps the user **build a stock portfolio** based on inputs like:

- Investment amount
- Time horizon or expected CAGR
- Sector preference (optional)
- Risk tolerance (beta threshold)

**How it works**:
- Scrapes financial ratios (CAGR, P/E, D/E) from ticker sites
- If sector unspecified, defaults to the **Nifty 50** basket
- Calculates **beta values** using P/E, P/B, and D/E
- Filters stocks based on **CAGR** and **beta**
- Picks **up to 3 stocks** (due to compute limits)
- Allocates weights based on relative CAGR
- Outputs selected stocks + weights to a CSV
- Results are wrapped and explained by an LLM layer

⚠️ **Avg. runtime: ~8 minutes** due to live scraping from non-API sources.

---

### 2️⃣ Sentiment Analyzer 📰

Scans financial news sites for **sentiment signals** about the user’s portfolio.

**Features**:
- Scrapes recent news articles on selected stocks
- Uses a general-purpose LLM for basic sentiment classification
- Flags stocks with **negative sentiment** and suggests review
- ⚠️ Does **not suggest replacements**—purely a warning system

---

### 3️⃣ RAG-Based Document Research 📚

Allows users to **ask questions about companies** based on scraped research PDFs.

**How it works**:
- Scrapes institutional research docs (PDFs) from public sources
- Uses `pdfplumber` to extract content
- Embeds paragraphs and chunks into a **vector database**
- LLM answers user queries with relevant citations

Current Limitations:
- Using `chromadb`, which has occasional retrieval bugs
- LLM is **LLaMA 3.2** — lightweight and fast, but accuracy is inconsistent

---

## ⚠️ Known Hiccups

- ⌛ **Slow execution** (~8 min/request) due to live scraping
- 💸 No paid APIs used (pure scraping), so data quality can vary
- 🧠 LLMs are fast but not deeply fine-tuned for finance
- 🧱 Vector store retrieval (RAG) is unstable—needs better infrastructure
- 🧾 PDF parsing sometimes misses tables and footnotes

---

## 🔮 Future Improvements

- Use **static data** for highly liquid stocks (e.g. Nifty50) with slow-changing ratios
- Replace `chromadb` with FAISS or Weaviate for stable RAG
- Optimize `pdfplumber` and switch to async scraping
- Integrate better sentiment classifier (finetuned FinBERT / financial RoBERTa)
- Create a Streamlit dashboard with interactive controls and visualizations

---

## 📁 Project Structure

