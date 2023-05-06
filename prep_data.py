import os
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, BertConfig, BertLMHeadModel, BertTokenizer


def load_image(image_path):
    return cv2.imread(image_path)

def threshold_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return thresh

def find_contours(thresh_image):
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_bounding_boxes(contours):
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h
        })
    return bounding_boxes

def save_bounding_boxes(bounding_boxes, output_file):
    with open(output_file, 'w') as f:
        json.dump(bounding_boxes, f)

def process_image(image_path, output_file):
    image = load_image(image_path)
    thresh_image = threshold_image(image)
    contours = find_contours(thresh_image)
    bounding_boxes = extract_bounding_boxes(contours)
    save_bounding_boxes(bounding_boxes, output_file)

# Set the input and output directories
input_dir = "data/images"
output_dir = "data/bounding_boxes"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# Iterate through images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        process_image(image_path, output_file)

if not os.path.exists('data/txt'):
    os.mkdir('data/txt')

# Load the feature extractor, tokenizer and captioning model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased", is_decoder=True)
model = BertLMHeadModel.from_pretrained("bert-base-uncased", config=config).to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process each image and save the corresponding caption
for img_filename in os.listdir('data/images'):
    # Process image here
    img = Image.open(os.path.join('data/images', img_filename)).convert('RGB')
    img = transform(img).unsqueeze(0)

    # Extract image features
    img_features = feature_extractor(img)['pixel_values'].to(device)

    # Generate caption
    input_ids = tokenizer.encode("The image depicts", return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    outputs = model(input_ids=input_ids, encoder_outputs=(img_features,), attention_mask=attention_mask)
    logits = outputs.logits
    predicted_caption = tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)

    # Save caption to file
    txt_filename = os.path.splitext(img_filename)[0] + '.txt'
    with open(os.path.join('data/txt', txt_filename), 'w') as f:
        f.write(predicted_caption[0])