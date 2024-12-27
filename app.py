import streamlit as st
from utils import load_document, create_knowledge_base, initialize_llm, initialize_qa_chain

def display_colored_output(question, response, retrieved_docs=None):
    st.markdown(f"<p style='color:red'>Question: {question}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue'>Response: {response}</p>", unsafe_allow_html=True)
    if retrieved_docs:
        with st.expander("Referenced Documents"):
            for i, doc in enumerate(retrieved_docs, 1):
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                st.markdown(f"<p style='color:green'>Document {i}: {preview}</p>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Project QA Chatbot", page_icon=":robot:")
    st.title("Project QA Chatbot")

    uploaded_file = st.file_uploader("Upload your text file", type=["txt"])

    if uploaded_file is not None:
        sections = load_document(uploaded_file)
        knowledge_base = create_knowledge_base(sections)
        llm = initialize_llm()
        qa_chain = initialize_qa_chain(llm, knowledge_base)

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "docs" not in st.session_state:  # Store retrieved docs
            st.session_state.docs = []

        # Display chat messages from history on app rerun
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    display_colored_output(
                        st.session_state.messages[i - 1]["content"],
                        message["content"],
                        st.session_state.docs[i // 2] if i // 2 < len(st.session_state.docs) else None,
                    )
                else:
                    st.markdown(message["content"])

        if question := st.chat_input("Enter your question here"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                result = qa_chain({"question": question})
                response = result['answer']
                retrieved_docs = result['source_documents']
                st.session_state.docs.append(retrieved_docs)  # Append new docs
                display_colored_output(question, response, retrieved_docs)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()