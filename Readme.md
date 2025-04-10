# LLM PDF Helper with RAG

LLM PDF Helper with RAG is a tool designed to enhance your interaction with PDF documents using advanced Retrieval-Augmented Generation (RAG) techniques powered by large language models (LLMs).

The files `WinRAR.pdf` (converted from `WinRAR.chm`) are included solely for testing purposes. They serve as example documents to demonstrate the functionality of the tool. These files are not required for the general use of the application and can be replaced with your own PDF or CHM files for embedding and querying.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/realJiaoKan/LLM-PDF-Helper-with-RAG
    ```
2. Navigate to the project directory:
    ```bash
    cd LLM-PDF-Helper-with-RAG
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Embedding Documents
To prepare the document embeddings:
```bash
python embedder.py
```

### Generating Answers
To generate answers based on user queries:
```bash
python generator.py
```

### Gradio Demo Interface
You can also use the Gradio-based web interface for embedding and generating answers interactively:
```bash
python demo.py
```
This will launch a web interface where you can upload PDF files, select models, and ask questions.

### Modifying `settings.py`

To customize the behavior of the tool, you can modify the `settings.py` file. This file contains configuration options, including the embedding model. For example adding a new sentence embedding model:

1. Open the `settings.py` file in a text editor.

2. Go find some model you like on Hugging Face, maybe you will like this: [Embedding Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

3. Add the path to your model in 'EMBED_MODEL_OPTIONS' and its dimension in 'EMBED_MODEL_OPTIONS_DIM'.

4. Enjoy!

## License

This project is licensed under the [MIT License](LICENSE).
