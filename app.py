


import gradio as gr
import os
import torch

from model import create_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# setup class names
class_names = ["chocolate_cake", "cup_cakes", "red_velvet_cake"]

# Model and transforms preparation

# Create model
model, transforms = create_model()

# load saved weights
model.load_state_dict(
    torch.load(
        f="effnetb2_cake.pth",
        map_location=torch.device("cpu")
    )
)

# create a predict function

def predict(img):

  start_time = timer()

  # Transform the target image and add a batch dimension
  img = transforms(img).unsqueeze(dim=0)

  # put model into eval mode
  model.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(model(img), dim=1)

  # create a prediction label and prediction probability dictionary
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # calculate the prediction time
  pred_time = round(timer() - start_time, 5)

  # return the prediction dictionary and prediction time
  return pred_labels_and_probs, pred_time


# Create gradio app
title = "Keki 🎂😋"
description = "This app uses a machine learning model to predict the type of cake from an uploaded image. Whether it's a chocolate cake, cupcake, or red velvet cake, simply upload a photo of a cake, and the app will classify it as one of the three variety with high accuracy. Perfect for dessert enthusiasts and bakers alike!"
article = "Created by RubyLawren"

example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil",),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
