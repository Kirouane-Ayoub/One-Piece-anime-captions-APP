# One-Piece Anime Captions APP

## Model Details : 

+ **Model Name**: Git-base-One-Piece
+ **Base Model**: Microsoft's *git-base* model
+ **Model Type**: Generative Image-to-Text (GIT)
+ **Fine-Tuned** On: *One-Piece-anime-captions* dataset
+ **Fine-Tuning Purpose**: To generate text captions for images related to the anime series **One Piece**

## Model Description : 
+ **Git-base-One-Piece** is a fine-tuned variant of Microsoft's **git-base** model, specifically trained for the task of generating descriptive text captions for images from the **One-Piece-anime-captions** dataset. **https://huggingface.co/ayoubkirouane/git-base-One-Piece**

+ The dataset consists of **856 {image: caption}** pairs, providing a substantial and diverse training corpus for the model.**https://huggingface.co/datasets/ayoubkirouane/One-Piece-anime-captions**

+ The model is conditioned on both CLIP image tokens and text tokens and employs a **teacher forcing** training approach. It predicts the next text token while considering the context provided by the image and previous text tokens.


![git_architecture](https://github.com/Kirouane-Ayoub/One-Piece-anime-captions-APP/assets/99510125/28a42bc3-d64b-4fe8-b665-78b445bea22b)


## Limitations : 
+ The quality of generated captions may vary depending on the complexity and diversity of images from the 'One-Piece-anime-captions' dataset.
+ The model's output is based on the data it was fine-tuned on, so it may not generalize well to images outside the dataset's domain.
Generating highly detailed or contextually accurate captions may still be a challenge.


## Model Usage :  

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-to-text", model="ayoubkirouane/git-base-One-Piece")
```

**or**

```python
# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("ayoubkirouane/git-base-One-Piece")
model = AutoModelForCausalLM.from_pretrained("ayoubkirouane/git-base-One-Piece")
```
## Gradio APP : 
```
pip install -r requirements.txt 
python app.py
```

**Demo** : **https://huggingface.co/spaces/ayoubkirouane/Git-base-One-Piece**


![Screenshot at 2023-09-23 16-11-25](https://github.com/Kirouane-Ayoub/One-Piece-anime-captions-APP/assets/99510125/1ee31693-8104-41a8-97ae-1855bed60576)
