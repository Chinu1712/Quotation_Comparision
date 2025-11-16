# =============================================================
# IMPORTS (FINAL & FIXED)
# =============================================================

from dotenv import load_dotenv
import streamlit as st
import os
import uuid
import tempfile
import faiss_cpu as faiss


# LangChain LLM Models
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + Vector Store
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


# =============================================================
# STREAMLIT SETUP
# =============================================================

load_dotenv()
st.set_page_config(page_title="Best Quotation Recommender", page_icon="💬")
st.header("💬 Best Quotation Recommender (FAISS Powered)")


# =============================================================
# FAISS INITIALIZATION — FIXED FOREVER
# =============================================================

@st.cache_resource
def create_or_load_faiss():
    """Create or load FAISS index safely."""
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        db = FAISS.load_local("quotation_faiss_db", embedding_model)
        return db, embedding_model

    except:
        dim = len(embedding_model.embed_query("hello"))
        index = faiss.IndexFlatL2(dim)

        db = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )

        return db, embedding_model


faiss_db, embedding_model = create_or_load_faiss()


# =============================================================
# UI TABS
# =============================================================

tabs = st.tabs(["About", "Chatbot", "History"])


# =============================================================
# ABOUT TAB
# =============================================================

with tabs[0]:
    st.subheader("About This App")
    st.write("""
    This AI-powered insurance quotation recommender features:

    ✔ FAISS Vector Database  
    ✔ Insurance-specific quotation detector  
    ✔ Automatic comparison (query optional)  
    ✔ Long-term searchable history  
    ✔ Parallel LLM evaluation  

    You can:
    - Upload quotations
    - Analyze best quote
    - Search old stored quotations  
    """)


# =============================================================
# CHATBOT TAB
# =============================================================

with tabs[1]:

    uploaded_files = st.file_uploader(
        "📄 Upload Insurance Quotation PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    user_query = st.text_area("💬 Enter procurement query (optional):")

    if st.button("🔍 Analyze"):

        # PDF is compulsory, query is optional
        if not uploaded_files:
            st.warning("Please upload at least one quotation PDF.")
            st.stop()

        if not user_query.strip():
            user_query = "Recommend the best insurance quotation based on premium, add-ons, and coverage."

        all_texts = []
        metadata_list = []

        # ---------------------------------------------------------
        # LOAD PDFS
        # ---------------------------------------------------------
        for f in uploaded_files:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            pages = loader.load_and_split()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(pages)

            text = " ".join([d.page_content for d in docs])
            all_texts.append(text)

            metadata_list.append({
                "supplier": f.name.replace(".pdf", ""),
                "file_name": f.name,
                "quotation_id": str(uuid.uuid4())
            })

            st.success(f"Loaded {len(pages)} pages from {f.name}")

        combined_text = "\n\n".join(all_texts)


        # ---------------------------------------------------------
        # INSURANCE QUOTATION CLASSIFIER — FIXED
        # ---------------------------------------------------------
        relevance_prompt = PromptTemplate(
            template="""
            You are a classifier.

            Identify if this text is an INSURANCE QUOTATION.

            A valid insurance quotation includes:
            - An insurance company name (TATA AIG, ICICI Lombard, etc.)
            - Policy type (Motor OD, Standalone OD, etc.)
            - IDV (Insured Declared Value)
            - Premium breakup (Basic OD, Add-ons, GST)
            - Final Premium amount
            - Vehicle details
            - Validity note / disclaimer

            If it fits, respond ONLY:
            YES

            If not, respond ONLY:
            NO

            TEXT:
            {document}
            """,
            input_variables=["document"]
        )


        # ---------------------------------------------------------
        # INSURANCE RECOMMENDATION ENGINE
        # ---------------------------------------------------------
        recommend_prompt = PromptTemplate(
            template="""
            You are an insurance procurement expert.

            USER QUERY:
            {query}

            QUOTATION CONTENT:
            {document}

            TASK:
            - DO NOT summarize.
            - Compare quotations on:
                * Final Premium
                * Add-on cover value
                * IDV value
                * Coverage benefits
                * Value-for-money score
            - Provide structured output:
                1. Factor-by-factor comparison
                2. Pros & cons
                3. Final recommendation

            """,
            input_variables=["query", "document"]
        )


        parser = StrOutputParser()

        # Models
        model_cls = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        model_eval = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b")

        chain = RunnableParallel(
            relevance=relevance_prompt | model_cls | parser,
            recommendation=recommend_prompt | model_eval | parser
        )

        results = chain.invoke({"document": combined_text, "query": user_query})


        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------
        if results["relevance"].strip() != "YES":
            st.error("❌ This document is NOT an insurance quotation.")
            st.stop()


        # ---------------------------------------------------------
        # STORE IN FAISS (NOW WORKS PERFECTLY)
        # ---------------------------------------------------------
        for text, meta in zip(all_texts, metadata_list):
            faiss_db.add_texts([text], metadatas=[meta])

        faiss_db.save_local("quotation_faiss_db")

        st.success("🧠 Quotation stored successfully in FAISS memory!")

        st.subheader("🧠 Best Quotation Recommendation:")
        st.write(results["recommendation"])


# =============================================================
# HISTORY TAB
# =============================================================

with tabs[2]:

    search_input = st.text_input("Search old quotations:")

    if st.button("🔎 Search"):

        if not search_input.strip():
            st.warning("Enter a keyword (e.g., premium, IDV, AIG).")
            st.stop()

        matches = faiss_db.similarity_search(search_input, k=5)

        if not matches:
            st.info("No matching quotations found.")
        else:
            for r in matches:
                with st.expander(f"📄 {r.metadata.get('file_name', 'Unknown')}"):
                    st.write("Supplier:", r.metadata.get("supplier"))
                    st.write("Quotation ID:", r.metadata.get("quotation_id"))
                    st.write(r.page_content[:800] + "…")

