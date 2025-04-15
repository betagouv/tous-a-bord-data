import os

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from utils.db_utils import get_postgres_cs


def setup_rag():
    # Configuration of the LangChain connection
    CONNECTION_STRING = get_postgres_cs()
    # Use of a French sentence-transformers model
    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/sentence-camembert-base"
    )
    # Configuration de Claude
    llm = ChatAnthropic(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model="claude-3-sonnet-20240229",
    )
    vectorstore = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="tarification_embeddings",
        pre_delete_collection=True,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


st.title("Chat with the transport data")

# Initialization of the message history in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display of the message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input zone for the question
if prompt := st.chat_input("Ask your question about transport fares..."):
    # Add the question to the history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display of the question
    with st.chat_message("user"):
        st.markdown(prompt)
    # Preparation of the answer
    with st.chat_message("assistant"):
        with st.spinner("Search in progress..."):
            try:
                # Initialization of the RAG system
                qa_chain = setup_rag()
                # Construction of the contextualized prompt
                full_prompt = f"""As a specialist in public transport,
                answer the following question based solely on the information
                available in the database: {prompt}
                If you cannot find the exact information, indicate it clearly.
                """
                # Getting the answer
                response = qa_chain({"query": full_prompt})
                answer = response["result"]
                # Display of the answer
                st.markdown(answer)
                # Add the answer to the history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                # Option to see the sources
                with st.expander("See the sources"):
                    for doc in response["source_documents"]:
                        st.markdown(
                            "**Source:** "
                            + doc.metadata.get("source", "Not specified")
                        )
                        st.markdown(
                            "**Extract:** " + doc.page_content[:200] + "..."
                        )
            except Exception as e:
                st.error(f"Erreur lors de la recherche : {str(e)}")

# Button to clear the history
if st.button("Clear the history"):
    st.session_state.messages = []
    st.rerun()
