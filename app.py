# =============================================================
# IMPORTS (FINAL)
# =============================================================

from dotenv import load_dotenv
import streamlit as st
import os
import uuid
import tempfile

# LLMs
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# PDF & Splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + ChromaDB
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# NEW Chroma Client for Python 3.13
import chromadb
from chromadb.config import Settings


# =============================================================
# STREAMLIT SETUP
# =============================================================
load_dotenv()
st.set_page_config(page_title="Best Quotation Recommender", page_icon="💬")
st.header("💬 Insurance Quotation Recommender (Chroma + RAG)")


# =============================================================
# VECTOR STORE INITIALIZATION (NEW WORKING VERSION)
# =============================================================
@st.cache_resource
def get_vectorstore():

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_dir = "/mount/data/chroma_store"

    # Create a persistent Chroma client (NEW WORKING API)
    client = chromadb.PersistentClient(path=persist_dir)

    # Create or load the collection
    vectordb = Chroma(
        client=client,
        collection_name="quotations",
        embedding_function=embedding_model
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
    st.subheader("About")
    st.write("""
    This system:
    - Uses ChromaDB (persistent vector DB) on Streamlit Cloud  
    - Detects valid motor insurance quotations  
    - Generates expert recommendations  
    - Stores embeddings for search  
    - Query is optional  
    """)


# =============================================================
# CHATBOT TAB
# =============================================================
with tabs[1]:

    st.subheader("Upload Insurance Quotation PDFs")

    uploaded_files = st.file_uploader(
        "📄 Upload quotation PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    user_query = st.text_area("💬 Enter procurement query (optional):")

    if st.button("🔍 Analyze"):

        # PDF compulsory
        if not uploaded_files:
            st.warning("Please upload at least one quotation PDF.")
            st.stop()

        if not user_query.strip():
            user_query = (
                "Recommend the best quotation based on premium, IDV, add-ons, "
                "and value for money."
            )

        all_texts = []
        metadata_list = []

        # Process PDFs
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            pages = loader.load_and_split()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            docs = splitter.split_documents(pages)
            text = " ".join(doc.page_content for doc in docs)

            all_texts.append(text)

            metadata_list.append({
                "supplier": f.name.replace(".pdf", ""),
                "file_name": f.name,
                "quotation_id": str(uuid.uuid4())
            })

            st.success(f"Loaded {len(pages)} pages from {f.name}")

        combined_text = "\n\n".join(all_texts)

        # Relevance Prompt
        relevance_prompt = PromptTemplate(
            template="""
            You are a document classifier.

            Identify if this text is an INSURANCE QUOTATION.

            A valid insurance quotation typically contains:
            - Insurance company name
            - Policy type (Motor OD, Standalone OD, etc.)
            - IDV (Insured Declared Value)
            - Premium breakup (OD, add-ons, GST)
            - Final premium amount
            - Vehicle details (model, registration, variant)
            - Validity note or disclaimer

            If valid, reply:
            YES

            Otherwise:
            NO

            TEXT:
            {document}
            """,
            input_variables=["document"]
        )

        # Recommendation Prompt
        recommend_prompt = PromptTemplate(
            template="""
            You are an insurance procurement expert.

            USER QUERY:
            {query}

            QUOTATION CONTENT:
            {document}

            Compare quotations based on:
            - Final Premium
            - IDV
            - Add-on covers
            - Value for money
            - Clarity of terms

            Provide:
            1. Factor-by-factor comparison  
            2. Pros & cons  
            3. Final recommendation  
            """,
            input_variables=["query", "document"]
        )

        parser = StrOutputParser()

        # Models
        model_classifier = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        model_recommender = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b"
        )

        # Parallel chain
        chain = RunnableParallel(
            relevance=relevance_prompt | model_classifier | parser,
            recommendation=recommend_prompt | model_recommender | parser
        )

        result = chain.invoke({
            "document": combined_text,
            "query": user_query
        })

        if result["relevance"].strip() != "YES":
            st.error("❌ This does not appear to be an insurance quotation.")
            st.stop()

        # Store in Chroma
        for text, meta in zip(all_texts, metadata_list):
            vectorstore.add_texts([text], metadatas=[meta])

        vectorstore.persist()

        st.success("🧠 Stored quotation in database!")

        st.subheader("🧠 Recommendation:")
        st.write(result["recommendation"])


# =============================================================
# HISTORY TAB
# =============================================================
with tabs[2]:

    st.subheader("Search Previous Quotations")

    q = st.text_input(
        "Search (e.g., 'TATA', 'premium', 'IDV', 'zero dep')"
    )

    if st.button("Search"):

        if not q.strip():
            st.warning("Enter a keyword.")
            st.stop()

        docs = vectorstore.similarity_search(q, k=5)

        if not docs:
            st.info("No matching quotations found.")
        else:
            for d in docs:
                meta = d.metadata or {}
                with st.expander(meta.get("file_name", "Unknown File")):
                    st.write("Supplier:", meta.get("supplier"))
                    st.write("Quotation ID:", meta.get("quotation_id"))
                    st.write(d.page_content[:800] + "…")
