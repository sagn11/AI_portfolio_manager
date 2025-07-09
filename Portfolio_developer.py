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
import time
import random
model=Ollama(model="llama3.2")
##Input Prompt template
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_template = """
You are an intelligent investment assistant.

Your job is to extract structured data from the user's natural language prompt.

**You must return a valid Python dictionary string, and nothing else.**

The dictionary should contain the following keys:

- "sector"
- "investment_type"
- "investment_goal"
- "risk_appetite"
- "investment_horizon" (as an integer number of years)
- "principal_amount" (in numeric format, no currency symbols)
- "final_amount" (in numeric format)
- "number_of_years"
- "growth_rate" (as a percentage, no % sign — leave blank if not specified)

**Constraints:**
- Only use these sectors: ["IT", "Banking & Finance", "Pharma", "Energy", "FMCG"]
- If a field is missing in the prompt, set its value to an empty string or `None` (but keep the key).
- You must return only the dictionary (as a single-line valid Python dictionary string). Ensure all keys are present, braces are properly closed, and the syntax is correct.

Example:
{{
    "sector": "Pharma",
    "investment_type": "lumpsum",
    "investment_goal": "wealth creation",
    "risk_appetite": "medium",
    "investment_horizon": 5,
    "principal_amount": 100000,
    "final_amount": 200000,
    "number_of_years": 5,
    "growth_rate": ""
}}
"""
##ticker names
sector_tickers = {
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT", "COFORGE", 
        "MPHASIS", "BSOFT", "SONATSOFTW", "CYIENT", "ZENSARTECH", "NIITLTD", 
        "KELLTONTEC", "TATAELXSI", "ECLERX", "NEWGEN", "INTELLECT", "HAPPSTMNDS"
    ],
    "Banking & Finance": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "YESBANK", 
        "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "RBLBANK", "PNB", "CANBK", "BANKBARODA", 
        "UNIONBANK", "AUBANK", "IDBI", "UJJIVANSFB", "CENTRALBK", "SOUTHBANK"
    ],
    "Pharma": [
        "SUNPHARMA", "DIVISLAB", "DRREDDY", "CIPLA", "LUPIN", "BIOCON", "TORNTPHARM", 
        "AUROPHARMA", "ZYDUSLIFE", "ALKEM", "GLAND", "IPCALAB", "PFIZER", 
        "ABBOTINDIA", "SANOFI", "NATCOPHARM", "GRANULES", "AJANTPHARM", 
        "JUBLPHARMA", "INDOCO"
    ],
    "Energy": [
        "RELIANCE", "ONGC", "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN", 
        "ADANITRANS", "NHPC", "GAIL", "OIL", "IOC", "BPCL", "HPCL", 
        "JSWENERGY", "SJVN", "TORNTPOWER", "CESC", "NLCINDIA", "BHEL", "COALINDIA"
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", 
        "GODREJCP", "COLPAL", "EMAMILTD", "VBL", "TATACONSUM", "UBL", 
        "RADICO", "ZYDUSWELL", "HATSUN", "KRBL", "MANAPPURAM", 
        "HERITGFOOD", "EVEREADY", "JYOTHYLAB"
    ]
}
nifty_50_comp=[
    "HDFCBANK",
    "ICICIBANK",
    "RELIANCE",
    "INFY",
    "BHARTIARTL",
    "ITC",
    "LT",
    "TCS",
    "AXISBANK",
    "KOTAKBANK",
    "SBIN",
    "M&M",
    "BAJFINANCE",
    "HINDUNILVR",
    "SUNPHARMA",
    "ULTRACEMCO",
    "WIPRO",
    "NTPC",
    "ASIANPAINT",
    "HCLTECH"
    ]

def beta_risk_level(risk):
    if risk=="low":
        return (0.0,0.8)
    if risk=="moderate":
        return (0.8,1.2)
    if risk=="high":
        return (1.2,2.5)
    if risk=="":
        return(0.8,1.2)
def calc_weights(dict):
    total_cagr = sum(data["5yr_CAGR"] for data in dict.values())
    weights_calculated = {
    ticker: data["5yr_CAGR"] / total_cagr
    for ticker, data in dict.items()
    }
    return weights_calculated
explanation_human_template = """
Original Input:
{input}

Selected Stocks:
{results}

Portfolio Weights:
{weights}
"""
explanation_system_template = """
You are a financial advisor assistant that explains an investment portfolio to the user in clear, professional, and engaging language.

You will be given:
- The user's original investment intent (as natural language)
- A dictionary of selected stocks with their financial metrics
- A dictionary of weights (allocation percentages)

Your job is to:
1. Summarize the user's investment preferences (sector, risk, growth, time).
2. List each selected stock along with its rationale for selection:
    - 5yr CAGR
    - ROE, ROCE
    - PE ratio
    - Beta (estimated)
    - Debt-to-equity
3. Explain why each stock fits the user's risk and return profile.
4. Clearly state the portfolio weights and how much of the principal should be invested in each.
5. End with a short paragraph about the portfolio's strengths and its alignment with the user's goals.

Be crisp, confident, and use easy-to-understand finance language. Keep it factual and structured, but not robotic.

Do NOT add disclaimers or legal text unless asked.
"""
def estimate_beta(pe=None, roce=None, roe=None, debt_to_equity=None, market_cap=None):
    # Normalize inputs to risk scale (0 to 1), based on common Indian market ranges
    risk = 0

    if pe is not None:
        risk += min(pe / 40, 1) * 0.25  # High P/E → growth → higher risk
    if roce is not None:
        risk += max(0, (20 - roce) / 20) * 0.25  # Lower ROCE → higher risk
    if roe is not None:
        risk += max(0, (20 - roe) / 20) * 0.15  # Lower ROE → more risk
    if debt_to_equity is not None:
        risk += min(debt_to_equity / 2, 1) * 0.25  # Higher leverage → more risk
    if market_cap is not None:
        if market_cap < 5000:       # Small cap
            risk += 0.1
        elif market_cap < 20000:    # Mid cap
            risk += 0.05

    # Map risk to beta range (0.6 to 2.0)
    beta = 0.6 + (1.4 * min(risk, 1.0))
    return round(beta, 2)

import pandas as pd

def save_portfolio_with_pandas(results, weights, filename="portfolio.csv"):
    data = []
    for ticker, metrics in results.items():
        entry = metrics.copy()  # Don't mutate original data
        entry['ticker'] = ticker
        entry['weight'] = round(weights.get(ticker, 0), 4)  # Safe access + rounding
        data.append(entry)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)  # <- Save the CSV file
    print(f"✅ Portfolio saved to {filename}")
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re

def get_finology_ratios(ticker):
    url = f"https://ticker.finology.in/company/{ticker}"
    
    options = Options()
    options.add_argument("--headless")  # Comment this line to see browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)

        # Wait for a known element to make sure the page is fully loaded
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class,'compess')][small[contains(text(),'P/E')]]")))

        # Get the current CAGR text to compare after click
        old_cagr_text = driver.find_element(By.ID, "pricereturn").text.strip()

        # Click the 5Yr CAGR button
        cagr_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-duration='5Yr']")))
        cagr_button.click()

        # Wait for CAGR value to change
        wait.until(lambda d: d.find_element(By.ID, "pricereturn").text.strip() != old_cagr_text)

        # Now fetch updated CAGR value
        cagr_text = driver.find_element(By.ID, "pricereturn").text.strip()

        # Helper functions
        def get_text(xpath):
            try:
                return driver.find_element(By.XPATH, xpath).text.strip()
            except:
                return None

        def parse_number(text):
            if not text:
                return None
            text = text.replace(',', '').replace('%', '').strip()
            try:
                return float(text)
            except:
                return None

        ratios = {}
        ratios['PE'] = parse_number(get_text("//div[contains(@class,'compess')][small[contains(text(),'P/E')]]/p"))
        ratios['ROCE'] = parse_number(get_text("//div[contains(@class,'compess')][small[contains(text(),'ROCE')]]//span[@class='Number']"))
        ratios['ROE'] = parse_number(get_text("//div[contains(@class,'compess')][small[contains(text(),'ROE')]]//span[@class='Number']"))
        ratios['Debt_Equity'] = parse_number(get_text("//div[@id='mainContent_divDebtEquity']//span[@class='Number']"))
        ratios['Market_Cap_Cr'] = parse_number(get_text("//p/span[@class='Number']"))

        match = re.search(r"([\d.]+)", cagr_text)
        ratios['5yr_CAGR'] = float(match.group(1)) if match else None

        return ratios

    except Exception as e:
        print(f"[!] Error scraping {ticker}: {e}")
        return None

    finally:
        driver.quit()
##result_dict contains the principal, years in a structured format
# Human/user prompt — this is where {input} goes

def portfolio_developer(input):
    human_template ="{input}"

    # Build the full ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    chain=prompt| model | StrOutputParser()
    result_dict_string = chain.invoke({"input": input})
    import ast
    print(result_dict_string)  # inspect this

    result_dict = ast.literal_eval(result_dict_string)


    principal=int(result_dict["principal_amount"])
    years=int(result_dict["investment_horizon"])
    risk=result_dict["risk_appetite"]
    sector=result_dict["sector"]
    if result_dict["growth_rate"]=="":
        final_amount=int(result_dict["final_amount"])
        growth_rate = (final_amount / principal) ** (1 / years) - 1
    else:
        growth_rate=float(result_dict["growth_rate"])
    if growth_rate < 10:
        print("⚠️ Warning: Growth rate too low for equity investing. Consider debt instruments instead.")
    
        print("Low growth rate detected. This may take longer as fewer companies match the criteria.")

    beta_low,beta_high=beta_risk_level(risk)
    if sector=="":
        companies=nifty_50_comp
    else:
        companies=sector_tickers[sector]
    results = {}
    c=0
    i=0


    while c < 4 and i < len(companies):
        slug = companies[i]
        

        data = get_finology_ratios(slug)
        try:
            PE = float(data["PE"])
            ROCE = float(data["ROCE"])
            ROE = float(data["ROE"])
            Market_cap = float(data["Market_Cap_Cr"])
            debt_equity = float(data["Debt_Equity"])
            cagr = float(data.get("5yr_CAGR", 0))
        except (ValueError, TypeError):
            i += 1
            continue

        beta_data = estimate_beta(pe=PE, roce=ROCE, roe=ROE, debt_to_equity=debt_equity, market_cap=Market_cap)
        print(f"\nCompany: {slug}")
        print(f"  Estimated Beta: {beta_data:.2f} | Max Acceptable beta :  {beta_high})")
        print(f"  Company 5yr CAGR: {data['5yr_CAGR']} | Target Growth Rate: {growth_rate:.2f} ± 6")
        if ( beta_data <= beta_high + 0.2) and (growth_rate - 6<= cagr <= growth_rate + 6):
            print(data)
            results[slug] = data
            c += 1
    
        i += 1
        time.sleep(random.uniform(2.5, 5.0))  # Respect Screener's server
    portfolio_weights=calc_weights(results)

    prompt2 = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(explanation_system_template),
        HumanMessagePromptTemplate.from_template(explanation_human_template),
    ])
    chain2=prompt2|model|StrOutputParser()
    output = chain2.invoke({
        "input": input,
        "results": results,
        "weights": portfolio_weights
    })
    
    save_portfolio_with_pandas(results, portfolio_weights, filename="portfolio.csv")
    return output
if __name__ == "__main__":
    user_prompt = input("🧠 Enter your investment intent:\n")
    portfolio_developer(user_prompt)
