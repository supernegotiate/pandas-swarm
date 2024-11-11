import pandas as pd
import streamlit as st
from swarmz import Agent


#################################################
### DATA LOADING AND ANALYSIS FUNCTIONS
#################################################


@st.cache_data
def load_data(file):
    """
    Load and cache CSV data using pandas.
    Args:
        file: CSV file object
    Returns:
        pandas DataFrame or None if error
    """
    try:
        return pd.read_csv(file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty")
    except pd.errors.ParserError:
        st.error("Unable to parse the CSV file. Please check the file format")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
    return None


def calculate_missing_values(df):
    """
    Calculate missing values statistics.
    Args:
        df: pandas DataFrame
    Returns:
        DataFrame with missing value statistics
    """
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_stats = pd.concat(
        [missing, missing_pct], axis=1, keys=["Missing Values", "Percentage"]
    )
    return missing_stats[missing_stats["Missing Values"] > 0].sort_values(
        "Percentage", ascending=False
    )


def data_overview(df):
    """Display sample of dataset"""
    st.header("Few rows of the dataset", divider=True)
    st.write(df.sample(n=min(5, len(df)), random_state=42))


def data_quality_check(df):
    """Display missing values analysis"""
    st.header("Missing Values of the dataset", divider=True)
    missing_table = calculate_missing_values(df)

    if not missing_table.empty:
        st.write(missing_table)
    else:
        st.write("No missing values found in the dataset.")


#################################################
### AGENT QUERY PROCESSING FUNCTIONS
#################################################


def save_audio_file(audio_value, file_path):
    """
    Save audio data to a file
    Args:
        audio_value: Audio data buffer
        file_path: Path to save the audio file
    """
    try:
        with open(file_path, "wb") as out_file:
            out_file.write(audio_value.getbuffer())
    except IOError as e:
        st.error(f"Error saving audio file: {str(e)}")


def process_analytics_query(swarm_client, question, df):
    """
    Process analytics query using swarm agent
    Args:
        swarm_client: Swarm client instance
        question: User's question
        df: pandas DataFrame
    Returns:
        Agent's response content
    """
    try:
        analytics_agent = Agent(
            name="Analytics",
            instructions=st.session_state["system_prompt"],
            model=st.session_state["analysis_model"],
        )

        user_query = f"""
            I have a dataframe containing below information: 
            ##dataframe##:
                {df}
            Please answer this question: {question}
        """

        response = swarm_client.run(
            agent=analytics_agent,
            context_variables={"dataframe": df},
            messages=[{"role": "user", "content": user_query}],
        )

        return response.messages[-1]["content"]

    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "Sorry, there was an error processing your query."


#################################################
### STREAMLIT UI CONFIGURATION
#################################################


def display_app_instructions():
    """Display instructions for using the app"""
    st.title(":material/speech_to_text: Speak With Data")
    with st.expander("### How to use app:"):
        st.markdown(
            """
            1. 🔑 Enter your [OpenAI API Key](https://beta.openai.com/account/api-keys) in the Settings Page
            2. 📁 In Dashboard page, upload your CSV file
            3. 📈 Explore Data Overview Tab
            4. 🎤 Ask your question from dataset using the microphone
            """
        )


def display_data_metrics(df):
    """Display key metrics about the dataset"""
    row_count, column_count = df.shape
    duplicates = df.duplicated().sum()
    metrics = st.columns(3)
    metrics[0].metric(label="Number of Rows", value=row_count)
    metrics[1].metric(label="Number of Columns", value=column_count)
    metrics[2].metric(label="Number of Duplicate Rows", value=duplicates)


def transcribe_audio(audio_path):
    """Transcribe audio file using OpenAI Whisper"""
    with open(audio_path, "rb") as audio_file:
        return (
            st.session_state["openai_client"]
            .audio.transcriptions.create(
                model="whisper-1",
                language="en",
                temperature=st.session_state["temperature"],
                prompt="Please transcribe the audio accurately, ignoring background noise and filler words.",
                file=audio_file,
            )
            .text
        )


def process_audio_question(df):
    """Handle audio input and process the question"""
    audio_value = st.audio_input("Ask your question from dataset")
    if audio_value:
        save_audio_file(audio_value, "audio.wav")
        transcription = transcribe_audio("audio.wav")
        st.info("Your Question: " + transcription, icon="🎤")
        with st.spinner("Generating Response..."):
            st.write(
                process_analytics_query(
                    st.session_state["swarm_client"], transcription, df
                )
            )


display_app_instructions()

uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        display_data_metrics(df)
        tab1, tab2, tab3 = st.tabs(["Data Overview", "Missing Values", "All Data"])
        with tab1:
            data_overview(df)
            process_audio_question(df)

        with tab2:
            data_quality_check(df)

        with tab3:
            st.header("All rows of the dataset", divider=True)
            st.write(df)
