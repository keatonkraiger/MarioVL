
from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

breakpoint()
txt = image_to_text('data/images/00000.jpg')

breakpoint()

print('hello')