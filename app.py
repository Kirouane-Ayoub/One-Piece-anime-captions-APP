import gradio as gr
from transformers import AutoProcessor,  AutoModelForCausalLM, pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint1 = "ayoubkirouane/git-base-One-Piece"
processor = AutoProcessor.from_pretrained(checkpoint1)
model1 = AutoModelForCausalLM.from_pretrained(checkpoint1)
def img2cap(image):
    input1 = processor(images=image, return_tensors="pt").to(device)
    pixel_values1 = input1.pixel_values
    generated_id1 = model1.generate(pixel_values=pixel_values1, max_length=50)
    generated_caption1 = processor.batch_decode(generated_id1, skip_special_tokens=True)[0]
    return generated_caption1


examples = ["1.png" ,"2.png" , "3.png" ]

gr.Interface(
    img2cap,
    inputs = gr.inputs.Image(type="pil", label="Original Image"),
    outputs= gr.outputs.Textbox(label="Caption from pre-trained model"),
    title = "Image Captioning using git-base-One-Piece Model",
    description = "git-base-One-Piece is used to generate Image Caption for the uploaded image.",
    examples=examples,
    theme="huggingface",
).launch(debug=True)
