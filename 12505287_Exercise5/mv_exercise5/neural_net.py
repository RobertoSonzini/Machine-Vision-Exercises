import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device

def predict_single_image(img_path, model, class_names, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_idx = torch.max(output, 1)
    
    return class_names[pred_idx.item()]

if __name__ == "__main__":
    model_path = "model_final.pt"
    class_names = ['book', 'cookiebox', 'cup', 'ketchup', 'sugar', 'sweets', 'tea']
    imgs_tests = ["imgs/steps/obj1.png", "imgs/steps/obj2.png", "imgs/steps/obj3.png", "imgs/steps/obj4.png", "imgs/steps/obj5.png", "imgs/steps/obj6.png"]

    model, device = load_model(model_path)

    print("Device in uso:", device)

    for img in imgs_tests:
        pred_class = predict_single_image(img, model, class_names, device)
        print("Predizione:", pred_class)
