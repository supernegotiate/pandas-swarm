import streamlit as st
from openai import OpenAI
from swarmz import Swarm

st.title(":material/settings: Application Settings")

api_key = st.text_input(
    "🔑 OpenAI API key",
    type="password",
    help="Your API key is not stored anywhere, and its valid just for current session",
)
st.warning(
    "Your API key is not stored anywhere, and its valid just for current session",
    icon="⚠️",
)

# Selection fields
left_col, center_col, right_col = st.columns(3)
analysis_model = left_col.selectbox(
    label="Select Model:",
    options=["gpt-4o", "gpt-4o-mini"],
    key="analysis_type",
)
embeddings_model = center_col.selectbox(
    label="Select Embedding:",
    options=["text-embedding-3-large", "text-embedding-3-small"],
    key="embeddings_model",
)

voice_model = right_col.selectbox(
    label="Select Voice Model:",
    options=["gpt-4o-audio-preview"],
    key="voice_model",
)

system_prompt = st.text_area(
    "System Prompt",
    """
                **You are an advanced data analysis assistant. Your role is to analyze datasets and provide clear, actionable insights based on user queries, especially regarding the data they upload. When given a dataset, you should:**

                1. **Understand the Dataset:**

                - Thoroughly assess its structure, including data types, content, and metadata.
                - Identify key features, categories, and relevant patterns that might assist in addressing the user's questions.

                2. **Perform Targeted Analysis:**

                - Analyze the data to uncover meaningful patterns, correlations, trends, and outliers relevant to the user's queries.
                - Apply appropriate statistical methods to gauge the significance of observed relationships.
                - Segment or categorize the data when necessary to provide comparative insights across different groups.

                3. **Deliver Focused Insights:**

                - Present final results and insights that directly respond to the user's questions, highlighting important statistics, correlations, and key findings.
                - Summarize key takeaways clearly and concisely, making the insights easily understandable and actionable.

                **Focus on delivering end results and insights rather than detailing intermediate analysis steps. Your goal is to provide the user with a straightforward understanding of the data based on their queries, not the process behind the analysis.**
            """,
    height=100,
)

st.write(f"You wrote {len(system_prompt)} characters.")


left_col2, center_col2, right_col2 = st.columns(3)

chunk_size = left_col2.number_input(
    "Chunk size (tokens)",
    1024,
    step=1024,
)
max_tokens: int = center_col2.number_input(
    "Max output (tokens)",
    1024,
    step=512,
)
temperature: float = right_col2.number_input(
    "Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format="%.1f"
)

if st.button("Update Config", type="primary", icon="💾"):
    client = OpenAI(api_key=api_key)
    swarm_client = Swarm(api_key=api_key)
    if "openai_api_key" not in st.session_state:
        st.session_state.update(
            {
                "openai_api_key": api_key,
                "analysis_model": analysis_model,
                "system_prompt": system_prompt,
                "chunk_size": chunk_size,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "swarm_client": swarm_client,
                "openai_client": client,
            }
        )
        st.success("Settings have been successfully updated.")
