import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# Load Gemini model
def get_gemini_model(model_name="gemini-pro"):
    return ChatGoogleGenerativeAI(model=model_name)

# Split large input into vectorized chunks
def embed_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragments = splitter.split_text(raw_text)
    
    embedding_engine = GooglePalmEmbeddings()
    db = Chroma.from_texts(fragments, embedding_engine)
    return db.as_retriever()

# Binary relevance scoring: yes/no
def filter_relevant_documents(query, docs):
    model = get_gemini_model()
    
    grading_prompt = ChatPromptTemplate.from_messages([
        ("human", "Retrieved document:\n\n{document}\n\nUser question:\n{question}\n\nIs this document relevant? (yes/no)")
    ])
    
    judge = grading_prompt | model | StrOutputParser()
    
    shortlisted = []
    for doc in docs:
        result = judge.invoke({"question": query, "document": doc.page_content})
        if "yes" in result.strip().lower():
            shortlisted.append(doc)
    
    return shortlisted

# Generate an answer based on selected context
def answer_from_documents(query, docs):
    model = get_gemini_model()
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("human", "Using this context:\n\n{context}\n\nAnswer the question:\n{question}")
    ])
    
    context_text = "\n\n".join([d.page_content for d in docs])
    retriever = embed_text_chunks(context_text)

    response_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": answer_prompt}
    )
    
    return response_chain.run(query)

# Full pipeline: extract → filter → answer
def crag_pipeline(question, source_text):
    retriever = embed_text_chunks(source_text)
    initial_docs = retriever.get_relevant_documents(question)
    
    filtered_docs = filter_relevant_documents(question, initial_docs)
    
    if not filtered_docs:
        # Optional: fallback method like web search or predefined answer
        from langchain_community.tools.tavily_search import TavilySearchResults
        alt_search = TavilySearchResults()
        web_fallback = alt_search.run(question)
        return f"No relevant context found.\n\nWeb search result:\n{web_fallback}"
    
    return answer_from_documents(question, filtered_docs)
