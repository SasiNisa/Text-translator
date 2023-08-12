# Importing the pipeline function from the transformers library 
from transformers import pipeline
import gradio as gr

# Creating a Text2TextGenerationPipeline for language translation 
pipe = pipeline(task='text2text-generation', model='facebook/m2m100_418M')

# Define the language dictionary outside the translate function
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Hindi": "hi"
}

def translate(text, target_language):
    target_lang_id = pipe.tokenizer.get_lang_id(lang=languages[target_language])
    translated_text = pipe(text, forced_bos_token_id=target_lang_id)
    return translated_text[0]['generated_text']
gr.close_all()

iface = gr.Interface(
    fn=translate,
    title="Text Translator", 
    inputs=[
        gr.inputs.Textbox(lines=2, label="Input Text"),
        gr.inputs.Dropdown(list(languages.keys()), label="Target Language")
    ],
    outputs=gr.outputs.Textbox(label="Translated Text")
)

iface.launch()

