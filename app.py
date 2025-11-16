# =============================================================
# IMPORTS
# =============================================================

from dotenv import load_dotenv
import streamlit as st
import os
import uuid
import tempfile

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

# Embeddings + Vector Store (Chroma)
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


# =============================================================
# STREAMLIT + ENV SETUP
# =============================================================

load_dotenv()
st.set_page_config(page_title="Best Quotation Recommender", page_icon="💬")
st.header("💬 Best Quotation Recommender (Chroma + RAG Enabled)")


# =============================================================
# VECTOR STORE (CHROMA) INITIALIZATION
# =============================================================

@st.cache_resource
def get_vectorstore():
    """
    Create or load a persistent Chroma vector store.
    Stored in /mount/data on Streamlit Cloud so it survives restarts.
    """
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_dir = "/mount/data/chroma_quotations"  # Streamlit Cloud writable path

    vectordb = Chroma(
        collection_name="quotations",
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )

    return vectordb, embedding_model


vectorstore, embedding_model = get_vectorstore()


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

    ✅ Chroma vector database (persistent on Streamlit Cloud)  
    ✅ Insurance-specific quotation classifier (motor OD quotes, etc.)  
    ✅ Query is optional – app can auto-recommend best quotation  
    ✅ Semantic history search over previously uploaded quotations  
    ✅ Parallel LLM pipeline (classifier + evaluator)  

    Upload 1 or more motor insurance quotation PDFs and get  
    a structured, factor-based recommendation.
    """)


# =============================================================
# CHATBOT TAB
# =============================================================

with tabs[1]:
    st.subheader("Upload Quotation PDFs")

    uploaded_files = st.file_uploader(
        "📄 Upload insurance quotation PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    user_query = st.text_area("💬 Enter your procurement query (optional):")

    if st.button("🔍 Analyze"):
        # ---------------------------
        # 1. PDF is MANDATORY
        # ---------------------------
        if not uploaded_files:
            st.warning("Please upload at least one quotation PDF.")
            st.stop()

        # ---------------------------
        # 2. Query is OPTIONAL
        # ---------------------------
        if not user_query.strip():
            user_query = (
                "Recommend the best insurance quotation based on premium, "
                "add-ons, IDV, coverage benefits, and overall value for money."
            )

        all_texts = []
        metadata_list = []

        # ---------------------------
        # 3. Load + Split PDFs
        # ---------------------------
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            pages = loader.load_and_split()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = splitter.split_documents(pages)
            text = " ".join(d.page_content for d in docs)
            all_texts.append(text)

            metadata_list.append({
                "supplier": f.name.replace(".pdf", ""),
                "file_name": f.name,
                "quotation_id": str(uuid.uuid4())
            })

            st.success(f"✅ Loaded {len(pages)} pages from **{f.name}**")

        combined_text = "\n\n".join(all_texts)

        # ---------------------------
        # 4. Relevance Classifier (Insurance-specific)
        # ---------------------------
        relevance_prompt = PromptTemplate(
            template="""
            You are a document classifier.

            Identify if this text is an INSURANCE QUOTATION.

            A valid insurance quotation usually includes:
            - Name of an insurance company (e.g., TATA AIG, Raheja QBE, ICICI Lombard)
            - Policy or quotation type (Motor OD, Standalone OD, Private Car Policy, etc.)
            - IDV (Insured Declared Value)
            - Premium breakup (Basic OD premium, add-ons, GST, Net Premium)
            - Final Premium amount
            - Vehicle details (model, registration no., variant, CC, etc.)
            - Disclaimer or validity note

            If the document resembles an insurance quotation, reply exactly:
            YES

            Otherwise reply exactly:
            NO

            TEXT:
            {document}
            """,
            input_variables=["document"],
        )

        # ---------------------------
        # 5. Recommendation Prompt
        # ---------------------------
        recommend_prompt = PromptTemplate(
            template="""
            You are an insurance procurement expert.

            USER QUERY:
            {query}

            QUOTATION CONTENT (may include multiple quotations):
            {document}

            TASK:
            - DO NOT summarize the document.
            - Compare quotations on:
                * Final premium (total cost including GST)
                * IDV (Insured Declared Value)
                * Add-on covers included (zero dep, RTI, engine protect, etc.)
                * Coverage benefits / value for money
                * Clarity and completeness of information
            - Provide output in this structure:

            1. Factor-by-Factor Comparison
               - Premium:
               - IDV:
               - Add-ons:
               - Coverage / Value for money:

            2. Pros & Cons
               - Pros of Quotation(s):
               - Cons / limitations:

            3. Final Recommendation
               - Clearly state which quotation is better and why, based ONLY on the document.
            """,
            input_variables=["query", "document"],
        )

        parser = StrOutputParser()

        # ---------------------------
        # 6. Models
        # ---------------------------
        model_classifier = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        model_recommender = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b"
        )

        # ---------------------------
        # 7. RunnableParallel: classify + recommend
        # ---------------------------
        chain = RunnableParallel(
            relevance=relevance_prompt | model_classifier | parser,
            recommendation=recommend_prompt | model_recommender | parser,
        )

        results = chain.invoke({"document": combined_text, "query": user_query})

        # ---------------------------
        # 8. Check if PDFs are valid quotations
        # ---------------------------
        if results["relevance"].strip() != "YES":
            st.error("❌ The uploaded document(s) do not look like insurance quotations.")
            st.stop()

        # ---------------------------
        # 9. Store in Chroma Vector DB
        # ---------------------------
        for text, meta in zip(all_texts, metadata_list):
            vectorstore.add_texts([text], metadatas=[meta])

        # Persist to disk so history survives restarts
        vectorstore.persist()

        st.success("🧠 Quotations stored in vector database successfully!")

        # ---------------------------
        # 10. Show Recommendation
        # ---------------------------
        st.subheader("🧠 Best Quotation Recommendation:")
        st.write(results["recommendation"])


# =============================================================
# HISTORY TAB – SEARCH PREVIOUS QUOTATIONS
# =============================================================

with tabs[2]:
    st.subheader("Search Stored Quotations")

    search_input = st.text_input(
        "🔎 Search by keyword (e.g., 'TATA', 'premium', 'IDV', 'zero dep'):"
    )

    if st.button("Search History"):
        if not search_input.strip():
            st.warning("Please enter a search keyword.")
            st.stop()

        # similarity_search returns a list of Documents
        docs = vectorstore.similarity_search(search_input, k=5)

        if not docs:
            st.info("No matching stored quotations found.")
        else:
            for doc in docs:
                meta = doc.metadata or {}
                with st.expander(f"📄 {meta.get('file_name', 'Unknown file')}"):
                    st.write("**Supplier:**", meta.get("supplier", "N/A"))
                    st.write("**Quotation ID:**", meta.get("quotation_id", "N/A"))
                    st.write(doc.page_content[:1000] + "…")
