import os
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import streamlit as st

def load_document(file):
    """Loads the text data from the uploaded file and splits it into sections."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    with open(tmp_file_path, 'r') as f:
        content = f.read()

    os.remove(tmp_file_path)

    sections = content.split('##########')
    return sections

def create_knowledge_base(sections):
    """Creates a FAISS knowledge base from the text sections."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=google_api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    knowledge_base = FAISS.from_texts(sections, embeddings)
    return knowledge_base

def initialize_llm():
    """Initializes the Groq language model."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0,
        groq_api_key=groq_api_key,
    )
    return llm

def initialize_qa_chain(llm, knowledge_base):
    """Initializes the Conversational Retrieval QA chain."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt_template = PromptTemplate.from_template('''
    You are a knowledgeable assistant for our service-based company XYZ, with access to relevant clients' project information. 
    Based on the given query and context, provide a concise, detailed response using the most relevant information available. 
    When discussing projects, please structure your response in the following format:

    Project Name: [Project Name]
    Project Overview: [Brief description of the project's goals and methods]
    Algorithms Tried: [List and briefly describe the algorithms used]
    Best Performing Algorithm: [State the best performing algorithm and why it was chosen]
    Key Metrics:
        - [Metric 1]: [Value]
        - [Metric 2]: [Value]
        - [Metric 3]: [Value]
        ...
    Next Steps: [Describe the planned next steps for the project]

    If the query asks about a general topic (e.g., "Have we done anything on Air Quality improvement?"), 
    try to identify and discuss multiple relevant projects in the above format, if available.
    If a relevant project is found but does not perfectly match the query, mention it briefly and explain how it relates to the query.
    If no directly relevant projects are found, provide a concise answer based on the available context and suggest potential next steps.

    Context: {context}
    Question: {question}

    Answer:
    ''')

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=knowledge_base.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain