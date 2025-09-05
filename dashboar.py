import streamlit as st

import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# --- Load Environment ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

# --- Constants ---
BASE_URL = "https://api.sectors.app/v1"
HEADERS = {"Authorization": SECTORS_API_KEY}

# --- Init LLM ---
llm = ChatGroq(
    temperature=0.7,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)


# ===================== UTILS ===================== #
def fetch_data(endpoint: str, params: dict = None):
    """Generic function to fetch data from Sectors API."""
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def run_llm(prompt_template: str, data: pd.DataFrame):
    """Format prompt with data and invoke LLM."""
    prompt = PromptTemplate.from_template(prompt_template).format(data=data.to_string(index=False))
    return llm.invoke(prompt).content


def clean_python_code(raw_code: str):
    """Cleans LLM-generated Python code block."""
    return raw_code.strip().strip("```").replace("python", "").strip()


# ===================== SECTIONS ===================== #
def sidebar_selector():

    """Sidebar untuk subsektor & perusahaan."""

    st.sidebar.title("📌 Pilihan Analisis")

    subsectors = fetch_data("subsectors/")
    subsector_list = pd.DataFrame(subsectors)["subsector"].sort_values().tolist()

    ## streamlit UI
    selected_subsector = st.sidebar.selectbox("🔽 Pilih Subsector", subsector_list)

    companies = fetch_data("companies/", params={"sub_sector": selected_subsector})
    companies_df = pd.DataFrame(companies)
    company_options = companies_df["symbol"] + " - " + companies_df["company_name"]

    ## streamlit UI
    selected_company = st.sidebar.selectbox("🏢 Pilih Perusahaan", company_options)

    return selected_company.split(" - ")[0]  # return symbol


def financial_summary(symbol: str):

    """Ringkasan eksekutif keuangan dari LLM."""

    financials = pd.DataFrame(fetch_data(f"financials/quarterly/{symbol}/",
                                         params={"n_quarters": "4",
                                                "report_date": "2023-09-30"}))

    prompt = """

    Anda adalah seorang analis keuangan yang handal.
    Berdasarkan data keuangan kuartalan berikut (dalam miliah Rupiah) :

    {data}

    Tuliskan ringkasan eksekutif dalam 3 poin singkat untuk seorang investor.
    Fokus pada:
    1. Tren pertumbuhan pendapatan (revenue)
    2. Tingkat profitabilitas
    3. Posisi arus kas operasi

    """
    summary = run_llm(prompt, financials)

    with st.expander("💡 Ringkasan Keuangan"):
        st.markdown(summary)

    return financials


def revenue_trend(symbol: str, financials: pd.DataFrame):

    """Generate line plot untuk tren pendapatan."""

    data_sample = financials[['date', 'revenue']].dropna()

    prompt = f"""

    Anda adalah seorang programer Python yang ahli dalam visualisasi data.

    Berikut adalah data pendapatan perusahaan:

    {data_sample}

    Buat sebuah skrip python menggunakan matplotlib untuk menghasilkan line plot.
    Instruksi :
    - Sumbu X adalah 'date'
    - Sumbu Y adalah 'revenue'

    Tulis HANYA kode python yang bisa langsung dieksekusi. Jangan sertakan penjelasan apapun.

    """
    code = clean_python_code(llm.invoke(prompt).content)

    with st.expander("📊 Visualisasi Tren Pendapatan"):
        exec_locals = {}
        exec(code, {}, exec_locals)
        #st.pyplot(exec_locals["fig"])


def trend_analysis(financials: pd.DataFrame):
    """Interpretasi tren keuangan (LLM)."""
    prompt = """

    Bertindaklah sebagai seorang analis keuangan.
    Berdasarkan data kuartalan berikut:
    {data}
    Analisis tren utama yang muncul dari data tersebut. fokus pada pergerakan revenue, net_income, dan operating.
    Sajikan analisis dalam 3 poin. Tuliskan dalam bahasa singkat, padat, jelas

    """
    analysis = run_llm(prompt, financials)
    with st.expander("🔎 Interpretasi Tren Keuangan"):
        st.markdown(analysis)


def risk_analysis(financials: pd.DataFrame):

    """Analisis risiko keuangan (LLM)."""

    prompt = """

    Anda adalah seorang analis risiko keuangan yang skeptis.
    Periksa data keuangan berikut dengan teliti:
    {data}
    identifikasi 2-3 potensi risiko atau "red flags" yang perlu diwaspadai dari data tersebut.
    Jelaskan dalam satu kalimat singkat

    """
    risks = run_llm(prompt, financials)
    with st.expander("⚠️ Potensi Risiko Keuangan"):
        st.markdown(risks)







# ===================== MAIN APP ===================== #
def main():
    symbol = sidebar_selector()

    if st.sidebar.button("🔍 Lihat Insight"):
        financials = financial_summary(symbol)
        revenue_trend(symbol, financials)
        trend_analysis(financials)
        risk_analysis(financials)


if __name__ == "__main__":
    main()
