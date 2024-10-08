
import torch
import torchvision
from torch import nn

def create_model(num_classes: int=3,
                 seed:int=42):
  # Create pretrained weightd, transforms and model
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.efficientnet_b2(weights=weights)

  # Freeze all layers in base model
  for param in model.parameters():
    param.requires_grad = False
  
  # Change clssifier head
  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features=1408, out_features=num_classes)
  )

  return model, transforms
