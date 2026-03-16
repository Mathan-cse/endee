import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data.txt","r") as f:
    documents = f.readlines()

doc_embeddings = model.encode(documents)

st.title("AI Semantic Search Project")

query = st.text_input("Enter search query")

if query:
    query_embedding = model.encode([query])
    similarities = np.dot(doc_embeddings, query_embedding[0])
    best_match = np.argmax(similarities)

    st.write("Best Match:")
    st.write(documents[best_match])