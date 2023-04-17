import streamlit as st
from functions.function import predict_response 

st.title("Welcome to Bank Chatbot")

with st.form("Chatbot", clear_on_submit=True):
    query = st.text_input("How can I help you?", placeholder="Type here...")
    submit = st.form_submit_button("Submit")
    
    if submit:
        st.write("**Question:**", query)
        st.write("**Answer:**", predict_response(query))
