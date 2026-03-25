import torch
from torchvision import transforms
from gesture_model import GestureClassifier

class GesturePredictor:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = config["classes"]
        self.model = GestureClassifier(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(config["gesture_model_path"], map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.threshold = config["confidence_threshold"]
        
        self.transform = transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, pil_image):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probs, 1)
            
            if max_prob.item() >= self.threshold:
                return self.classes[predicted.item()], max_prob.item()
            return None, max_prob.item()
