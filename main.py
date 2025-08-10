import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

# 🔧 Ustawienia
batch_size = 4
epochs = 24
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 
                      'cpu')

print("Using ", device)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=10),  # delikatne skręcenie
    transforms.ToTensor()
])

# 📥 Dane
dataset = ImageFolder(root='./assets/train', transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_data  = ImageFolder(root='./assets/test', transform=transform)
test_loader  = DataLoader(test_data, batch_size=batch_size)

# 🧠 Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        x = torch.randn(1, 3, 512, 512)  # 1 obraz RGB 512x512
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        self._flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self._flattened_size, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # (8, 16, 16)
        x = F.max_pool2d(x, 2)          # (8, 8, 8)
        x = F.relu(self.conv2(x))       # (16, 8, 8)
        x = F.max_pool2d(x, 2)          # (16, 4, 4)
        x = x.view(x.size(0), -1)      # flatten
        x = F.relu(self.fc1(x))         # (64,)
        x = self.fc2(x)                 # (10,)
        return x


from PIL import Image
import torch.nn.functional as F

def predict_image(image_path, model, class_names, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')  # gwarancja 3 kanałów
    img_tensor = transform(image).unsqueeze(0).to(device)  # dodaj batch_dim: (1, 3, H, W)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    print(f"\n✅ Predykcja: {class_names[pred_idx]} ({confidence*100:.1f}%)")
    print(f"🔍 Prawdopodobieństwa: {[(class_names[i], round(p.item()*100, 1)) for i, p in enumerate(probs[0])]}")

    # (opcjonalnie) wyświetl zdjęcie
    plt.imshow(image)
    plt.title(f"{class_names[pred_idx]} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.show()







model = SimpleCNN().to(device)

# ⚙️ Loss i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("CLASS MAPPING:", dataset.class_to_idx)


# 🚀 Trening
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")






# ✅ Test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).long()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")






# 👁️‍🗨️ Przykładowe predykcje
# matplotlib.use('TkAgg')

images, labels = next(iter(test_loader))
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

plt.figure(figsize=(10, 3))
for i in range(batch_size):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].cpu().permute(1, 2, 0))  # z (C,H,W) → (H,W,C)
    plt.title(f"Label: {labels[i]}\nPred: {predicted[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# predict_image(
#     image_path='./sabrina.jpg',
#     model=model,
#     class_names=dataset.classes,  # ['ari', 'max']
#     transform=transform,     # ten sam co do test_loadera
#     device=device
# )