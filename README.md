# RAG With Multi Data Source

## Introduction
This project demonstrates the use of the Retrieval-Augmented Generation (RAG) model with multiple data sources. It combines information from Wikipedia, PDF documents, web pages, and ArXiv papers to generate responses to user queries.

## Setup
1. **Installation**: Make sure you have Python installed on your system.
2. **Dependencies**: Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
3. **Environment Variables**: Set up the necessary environment variables, including your OpenAI API key.
## Usage
Running the Code: Execute the main script to run the RAG model with multi-data sources:
```bash
   main.py
```
Input: Enter your query when prompted.

Output: The program will generate a response based on the input query and the available data sources.

## File Structure
1. **main.py**: Main script to run the RAG model.
2. **requirements.txt**: List of Python dependencies.
3. **keec102.pdf**: Sample PDF document used for demonstration.
4. **README.md**: This file, providing an overview of the project.
## Acknowledgements
1. This project uses LangChain, an open-source library for natural language processing and generation.
2. The RAG model is based on research by Lewis et al. (2020): Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
