import streamlit as st
import requests

st.title("PDF Agents")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask something..."):
    # Append user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Make request to /chat endpoint
    response = requests.post("http://172.19.104.12:9169/chat", json={"query": user_input})
    print(response)
    if response.status_code == 200:
        bot_reply = response.json().get("response", "No response from server")
    else:
        bot_reply = "Error: Could not reach the server."
    
    # Append bot response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
