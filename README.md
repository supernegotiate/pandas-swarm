# GenAI Voice-Interactive Data Analytics App

This application provides a seamless voice interaction experience with a large language model (LLM) to analyze data from uploaded CSV files. By integrating multiple components, it offers an interactive and intuitive solution for deriving insights from structured data through voice commands.

## Key Features

- **LLM-Powered Analysis**  
  OpenAI serves as the core language model, processing inputs and providing insights based on extensive language understanding and contextual knowledge.

- **Speech-to-Text Conversion**  
  Utilizing OpenAI’s Whisper model, the app converts spoken input into text, enabling a smooth voice-driven interaction with the LLM.

- **User-Friendly Interface**  
  Built on the Streamlit framework, the app provides an intuitive frontend for users to interact with, whether they’re uploading CSV files, issuing voice commands, or viewing analytic results.

- **Data Insight Generation**  
  Users can upload CSV files and receive analytical summaries and insights powered by the LLM, turning complex data into easy-to-understand insights.

## Heart of the GenAI Framework: Swarm

This project is powered by the **Swarm** framework, a toolkit designed for building ergonomic interfaces for multi-agent systems, making agent coordination and execution lightweight, highly controllable, and easily testable. Swarm enables the development of complex, scalable solutions by using primitive abstractions, such as:

- **Agents**  
  Each agent encompasses instructions and tools, which makes it possible to represent intricate dynamics between tools and networks of agents. This approach provides a lightweight, scalable, and customizable framework suited for managing large networks of independent capabilities and instructions that can't be condensed into a single prompt.

### Customization Highlights

Our customized Swarm setup includes:

1. **Single-File Simplicity**  
   The core functionality is encapsulated in a single file with less than 390 lines of code, ensuring simplicity and ease of use.

2. **Support for Pandas DataFrames**  
   Integrates Pandas data frames as context variables, making data handling in analytics straightforward.

3. **Support for Base64 Encoding**  
   Enables passing any file as a context variable using Base64 encoding, providing flexibility for file management.

4. **Compatibility with Multiple LLMs**  
   Works with any LLM model compatible with the OpenAI API, including Ollama, Mistral, llama3.2, and more, allowing flexibility in model choice.  

## Technology Stack and Features

    - ✨ [Openai](https://openai.com/) for GenAI services.
    - 🐝 [OpenAI Swarm](https://www.langchain.com/) for LLM infrastructure.
    - 🦜 [Streamlit](https://streamlit.io/) for the frontend.
    - 🧰 [Pandas](https://pandas.pydata.org/) for data analysis and manipulation.
    - 🔍 [Pydantic](https://docs.pydantic.dev) for the data validation.
    - 🐋 [Docker](https://www.docker.com) for development and production.

## Quick start

### Run the app in containers

-   Clone the repo and navigate to the root folder.

-   To run the app using Docker, make sure you've got DockerCLI installed on your
    system. From the project's root directory, run:
    ```sh
    docker build -t z-ask-data .
    docker run -d -p 8501:8501 --name ask-data z-ask-data
    docker ps
    ```
### Or, run the app locally

If you want to run the app locally, without using Docker, then:

-   Install Python, version 3.12

-   Clone the repo and navigate to the root folder.

-   Create a virtual environment.
    Run:

    ```sh
    python3.12 -m venv .venv
    ```

-   Activate the environment. Run:

    ```sh
    # Windows
    ./venv/scripts/activate
    
    # macOS or Linux
    source ./venv/bin/activate
    ```


-   Install the dependencies. Run:

    ```bash
    pip install -r requirements.txt 
    ```

-   Start the app. Run:

    ```bash
    streamlit run app.py
    ```
-   Go to the following link on your browser:

    ```
    http://localhost:8501
    ```   

## Linter and Code formatter
### Ruff is extremely fast Python linter and code formatter, written in Rust.
-   Install :

    ```bash
    # On macOS and Linux.
    curl -LsSf https://astral.sh/ruff/install.sh | sh 

    # On Windows.
    powershell -c "irm https://astral.sh/ruff/install.ps1 | iex"
    ```

-   Once installed, you can run Ruff from the command line:

    ```bash
    ruff check   # Lint all files in the current directory.
    ruff format  # Format all files in the current directory.
    ```



