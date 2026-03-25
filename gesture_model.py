import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class GestureClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(GestureClassifier, self).__init__()
        # Use MobileNetV2 as a lightweight model
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Replace classifier for custom classes
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
    def forward(self, x):
        return self.model(x)
