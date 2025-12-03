

from dotenv import load_dotenv
import streamlit as st
import os
import uuid
import tempfile

# LLM Models
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# PDF Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma




load_dotenv()
st.set_page_config(page_title="Best Insurance Quotation Recommender", page_icon="💬")
st.header("💬 Best Quotation Recommender")


#vector store 

@st.cache_resource
def get_vectorstore():

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    #Chroma Db
    vectordb = Chroma(
        collection_name="quotations",
        embedding_function=embedding_model,
        
    )

    return vectordb, embedding_model


vectorstore, embedding_model = get_vectorstore()



tabs = st.tabs(["About", "Chatbot", "History"])


#ABOUT TAB
with tabs[0]:
    st.subheader("About This App")
    st.write("""
    This AI system:
    - Classifies whether PDFs are valid insurance quotations  
    - Extracts & evaluates premium, IDV, add-ons, coverage  
    - Makes a structured recommendation  
    - Stores quotations in an in-memory vector DB  
    - Allows history search during the session  

    Perfect for ValueMomentum demonstration.
    """)


#CHATBOT TAB

with tabs[1]:

    st.subheader("Upload Quotation PDFs")

    uploaded_files = st.file_uploader(
        "📄Upload quotation PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    user_query = st.text_area("💬 Enter your query (optional):")

    if st.button("🔍 Analyze"):

        # PDF compulsory
        if not uploaded_files:
            st.warning("Please upload at least one quotation PDF.")
            st.stop()

        # Query 
        if not user_query.strip():
            user_query = (
                "Recommend the best quotation based on premium, IDV, add-ons, "
                "coverage benefits, and value for money."
            )

        all_texts = []
        metadata_list = []

        # Load + Split PDFs
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
            text = " ".join(d.page_content for d in docs)

            all_texts.append(text)

            metadata_list.append({
                "supplier": f.name.replace(".pdf", ""),
                "file_name": f.name,
                "quotation_id": str(uuid.uuid4())
            })

            st.success(f"Loaded {len(pages)} pages from {f.name}")

        combined_text = "\n\n".join(all_texts)

        # Relevance Classifier
        relevance_prompt = PromptTemplate(
            template="""
            You are a classifier.

            Determine if this document is an INSURANCE QUOTATION.

            Valid quotation usually includes:
            - Insurance company name
            - Policy type
            - IDV value
            - Premium breakup (OD, add-ons, GST)
            - Final premium
            - Vehicle details
            - Validity note

            If valid: YES
            Else: NO

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

            QUOTATION DATA:
            {document}

            Compare quotations on:
            - Premium
            - IDV
            - Add-ons
            - Value for money

            Provide:
            1. Comparison  
            2. Pros / Cons  
            3. Final Recommendation  
            """,
            input_variables=["query", "document"]
        )

        parser = StrOutputParser()

        # Models
        model_cls = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        model_eval = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b"
        )

        # Parallel Chaining 
        chain = RunnableParallel(
            relevance=relevance_prompt | model_cls | parser,
            recommendation=recommend_prompt | model_eval | parser
        )

        result = chain.invoke({
            "document": combined_text,
            "query": user_query
        })

        # Validate
        if result["relevance"].strip() != "YES":
            st.error("❌ This does not appear to be an insurance quotation.")
            st.stop()

        # Store in vectorstore
        for text, meta in zip(all_texts, metadata_list):
            vectorstore.add_texts([text], metadatas=[meta])

        st.success("🧠 Stored quotations in memory!")

        st.subheader("🧠 Recommendation:")
        st.write(result["recommendation"])


# HISTORY TAB
with tabs[2]:

    st.subheader("Search Stored Quotations")

    q = st.text_input("Search (e.g., 'premium', 'TATA', 'IDV'): ")

    if st.button("Search"):

        if not q.strip():
            st.warning("Enter search keyword.")
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




