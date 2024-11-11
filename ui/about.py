import streamlit as st

# --- Header ---
st.header("Speak with your data 🔥!", anchor=False)

# --- Description ---
st.markdown(
""" 
This application provides a seamless voice interaction experience with a large language model (LLM) to analyze data from uploaded CSV files.
By integrating multiple components, it offers an interactive and intuitive solution for deriving insights from structured data through voice commands.
"""
)
# --- Usage ---
st.write("\n")
st.subheader("How to use app", anchor=False)
st.markdown(
    """
    1. 🔑 Enter your [OpenAI API Key](https://beta.openai.com/account/api-keys) in the Settings Page
    2. 📁 In Dashboard page, upload your CSV file (CSV Sample available in: 'sample_data' folder)
    3. 📈 Explore Data Overview Tab
    4. 🎤 Ask your question from dataset using the microphone
    """
)
# --- Technologies ---
st.write("\n")
st.subheader("Technologies", anchor=False)
st.write(
    """
    - ✨ [Openai](https://openai.com/) for GenAI services.
    - 🐝 [Customized OpenAI Swarm](https://www.langchain.com/) for LLM infrastructure.
    - 🦜 [Streamlit](https://streamlit.io/) for the frontend.
    - 🧰 [Pandas](https://pandas.pydata.org/) for data analysis and manipulation.
    - 🔍 [Pydantic](https://docs.pydantic.dev) for the data validation.
    - 🐋 [Docker](https://www.docker.com) for development and production.
    """
)
