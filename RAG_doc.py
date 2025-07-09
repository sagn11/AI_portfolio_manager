from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores  import Chroma
from langchain_community.llms import ollama
import requests
from bs4 import BeautifulSoup
import os
import re
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema.output_parser import StrOutputParser
import os
import re
import time
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings



def download_latest_research_report(company_ticker, save_dir="research_reports"):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://ticker.finology.in/company/{company_ticker.upper()}"

    try:
        html = requests.get(url, headers=headers).text
    except Exception:
        return None

    soup = BeautifulSoup(html, "html.parser")

    try:
        title_text = soup.title.text
        company_name = title_text.split("|")[0].strip()
    except:
        company_name = company_ticker.upper()

    os.makedirs(save_dir, exist_ok=True)

    report_list = soup.select("ul.reportsli li")
    latest_date = None
    latest_link = None
    latest_source = None

    for item in report_list:
        badge = item.select_one(".badge-research")
        if badge:
            anchor = item.find("a", href=True)
            date_text = item.select_one("small.text-grey")
            if not anchor or not date_text:
                continue

            try:
                date = datetime.strptime(date_text.text.strip(), "%d %b %Y")
            except:
                continue

            if (latest_date is None) or (date > latest_date):
                latest_date = date
                latest_link = anchor['href']
                latest_source = anchor.text.strip().replace("Report By:", "").replace("Report by:", "").strip()

    if not latest_link:
        return None

    safe_name = re.sub(r'\W+', '_', company_name)
    source_tag = re.sub(r'\W+', '_', latest_source)
    filename = f"{safe_name}_Research_{latest_date.strftime('%Y')}_{source_tag}.pdf"
    filepath = os.path.join(save_dir, filename)

    try:
        response = requests.get(latest_link, timeout=15)
        if "application/pdf" in response.headers.get("Content-Type", "").lower():
            with open(filepath, "wb") as f:
                f.write(response.content)
            return filename
        else:
            return None
    except:
        return None
import os
import re
import pdfplumber

def extract_text_from_pdf(pdf_path, save_txt=False, txt_output_dir="txt_outputs"):
    if not os.path.exists(pdf_path):
        print(f"[!] File not found: {pdf_path}")
        return None

    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        text = text.strip()

        if save_txt:
            os.makedirs(txt_output_dir, exist_ok=True)
            raw_name = os.path.basename(pdf_path).replace(".pdf", "")
            safe_name = re.sub(r"\W+", "_", raw_name)
            txt_filename = f"{safe_name}.txt"
            txt_path = os.path.join(txt_output_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as out:
                out.write(text)
            print(f"✅ Saved extracted text to: {txt_path}")
            return txt_filename

        return text

    except Exception as e:
        print(f"[!] Failed to extract from {pdf_path}: {e}")
        return None


def RAG(query, force=False):
    start = time.time()
    print("⏱ RAG pipeline started...")

    # 1. Extract company name
    system_prompt = """
    You are an assistant that extracts company names or stock tickers from user questions.
    Return only the ticker/company name, don't explain anything.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{query}")
    ])

    llm = Ollama(model="llama3.2")
    chain = prompt | llm | StrOutputParser()
    company_name = chain.invoke({"query": query}).strip()
    company_name = re.sub(r"\.(NS|BO|NSE|BSE)$", "", company_name.strip(), flags=re.IGNORECASE)
    company_ticker = re.sub(r'\W+', '', company_name).upper()
    print(f"🔍 Extracted company: {company_name} → Ticker: {company_ticker} [{time.time() - start:.2f}s]")

    # 2. Vector DB path
    db_dir = os.path.abspath(f"vector_db/{company_ticker}")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(db_dir) and len(os.listdir(db_dir)) > 0 and not force:
        print(f"📦 Using cached vector DB at {db_dir}")
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    else:
        print(f"📄 Rebuilding vector DB for {company_name}...")

        # Download and extract
        filename = download_latest_research_report(company_ticker, save_dir="reports")
        if not filename:
            return f"[!] Couldn't get report for '{company_name}'"

        pdf_path = os.path.join("reports", filename)
        txt_file = extract_text_from_pdf(pdf_path, save_txt=True)

        if not txt_file:
            return f"[!] PDF extraction failed: {filename}"

        txt_file = os.path.basename(txt_file).strip()
        if not txt_file.endswith(".txt"):
            txt_file += ".txt"

        txt_path = os.path.abspath(os.path.join("txt_outputs", txt_file))
        if not os.path.isfile(txt_path):
            return f"[!] Text file missing at {txt_path}"

        # Chunking
        docs = TextLoader(txt_path, encoding="utf-8").load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Embedding and DB creation
        db = Chroma.from_documents(chunks, embeddings, persist_directory=db_dir)
        db.persist()
        print(f"✅ Vector DB saved at {db_dir}")

    # 3. Retrieval + DEBUG: show top 3 chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})
    top_docs = retriever.get_relevant_documents(query)
    print("\n🔎 Top Retrieved Chunks:")
    for i, doc in enumerate(top_docs):
        print(f"\n--- Chunk {i+1} ---\n{doc.page_content[:800]}\n")

    # 4. QA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa_chain.run(query)

    print(f"✅ Done in {time.time() - start:.2f}s")
    return response
