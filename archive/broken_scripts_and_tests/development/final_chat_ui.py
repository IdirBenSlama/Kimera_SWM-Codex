import streamlit as st
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
KIMERA_API_URL = "http://127.0.0.1:8001/api/chat/"
st.set_page_config(page_title="KIMERA Chat", layout="wide")

# --- UI Components ---
st.title("🧠 KIMERA Universal Translator")
st.caption("A direct interface to the cognitive core.")

if 'session_id' not in st.session_state:
    st.session_state.session_id = "streamlit-chat-session"
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and API Call ---
if prompt := st.chat_input("What is your query?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request for KIMERA backend
    payload = {
        "message": prompt,
        "session_id": st.session_state.session_id,
        "mode": "natural_language"
    }

    try:
        # Call the API
        with st.spinner("KIMERA is thinking..."):
            response = requests.post(KIMERA_API_URL, json=payload, timeout=90)
            response.raise_for_status()  # Raise an exception for bad status codes

        # Process the response
        api_response = response.json()
        kimera_reply = api_response.get("response", "No response content found.")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": kimera_reply})
        with st.chat_message("assistant"):
            st.markdown(kimera_reply)

    except requests.exceptions.RequestException as e:
        error_message = f"Connection to KIMERA backend failed: {e}"
        st.error(error_message)
        logger.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        st.error(error_message)
        logger.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with instructions
st.sidebar.header("How to Use")
st.sidebar.info(
    """
    1.  **Stop Previous Processes:** Make sure no other `python` processes are running that might be using port `8001`.
    2.  **Run the Backend:** In your terminal, run the following command from the project root:
        ```bash
        python -m backend.api.main
        ```
    3.  **Run the Chat UI:** Once the backend is running, open a *new* terminal and run this command:
        ```bash
        streamlit run final_chat_ui.py
        ```
    4.  **Interact:** Type your questions in the chat box and press Enter.
    """
) 