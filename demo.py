import gradio as gr
from settings import EMBED_MODEL_OPTIONS, OPENAI_MODEL_OPTIONS

from embedder import Embedder
from generator import Generator


def embed(embed_model_name, pdf_file):
    if not pdf_file:
        yield gr.update(value="No PDF uploaded! Please try again.", interactive=True), gr.update(interactive=True)
        return

    try:
        yield gr.update(value="Embedding...", interactive=False), gr.update(interactive=False)
        embedder = Embedder(embed_model_name, pdf_file.name)
        embedder.embed()
        yield gr.update(value="Embed!", interactive=True), gr.update(interactive=True)
    except Exception as e:
        print(f"Error during embedding: {e}")
        yield gr.update(value="Error occurred! Please try again.", interactive=True), gr.update(interactive=True)

    return


def generate(openai_model_name, embed_model_name, temperature, user_query):
    try:
        generator = Generator(openai_model_name, embed_model_name, temperature)
        top_docs, prompt, answer = generator.generate(user_query)
        table_data = [[doc["score"], doc["text"]] for doc in top_docs]
        return answer, table_data
    except Exception as e:
        print(f"Error during generation: {e}")
        return "An error occurred during generation. Please try again.", []


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            pdf_file = gr.File(label="Upload PDF File")
            embed_model_dropdown = gr.Dropdown(
                label="Embedding Model",
                choices=EMBED_MODEL_OPTIONS,
                value=EMBED_MODEL_OPTIONS[0]
            )
            embed_button = gr.Button("Embed!")
            openai_model_dropdown = gr.Dropdown(
                label="OpenAI Model",
                choices=OPENAI_MODEL_OPTIONS,
                value=OPENAI_MODEL_OPTIONS[0]
            )
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.1,
                label="Temperature"
            )
        with gr.Column():
            user_query = gr.Textbox(label="Enter your query")
            submit_button = gr.Button("Generate")
            output_box = gr.Textbox(label="Answer")
            top_docs_table = gr.Dataframe(
                headers=["Score", "Doc Found"],
                label="Top Docs",
                wrap=True
            )

    embed_button.click(
        embed,
        inputs=[embed_model_dropdown, pdf_file],
        outputs=[embed_button, submit_button]
    )

    submit_button.click(
        generate,
        inputs=[
            openai_model_dropdown,
            embed_model_dropdown,
            temperature_slider,
            user_query
        ],
        outputs=[output_box, top_docs_table]
    )

demo.queue()
demo.launch()
