import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = [SimpleCNN(), DeeperCNN(), AdvancedCNN()]
model_names = ["Standardowa", "Rozbudowana", "Architektura z barchNorm2d"]

# for model, model_name in zip(models, model_names):
#     print(f"Training {model_name}...")
#
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # Trening modelu
#     num_epochs = 5
#     train_losses = []
#     test_losses = []
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         train_losses.append(running_loss / len(train_loader))
#         print(f"Liczba epok [{epoch+1}/{num_epochs}], Strata: {running_loss/len(train_loader):.4f}")
#
#     model.eval()
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
#     incorrect_images = []
#
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#             for i in range(len(labels)):
#                 if predicted[i] != labels[i]:
#                     incorrect_images.append((images[i], predicted[i], labels[i]))
#
#     accuracy = 100 * correct / total
#     print(f"Accuracy of the model on the test images: {accuracy:.2f}%")
#
#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
#     plt.xlabel('Przewidziane wartości')
#     plt.ylabel('Prawdziwe wartości')
#     plt.title(f'Macierz pomyłek - {model_name}')
#     plt.show()
#
#     plt.plot(range(1, num_epochs + 1), train_losses, label=f'Train Loss - {model_name}')
#     plt.xlabel('Epoki')
#     plt.ylabel('Strata')
#     plt.title(f'Strata dla modelu - {model_name}')
#     plt.legend()
#     plt.show()
#
#     incorrect_image, predicted_class, true_class = incorrect_images[0]
#     incorrect_image = incorrect_image.squeeze().cpu().numpy()
#     plt.imshow(incorrect_image, cmap='gray')
#     plt.title(f'Przewidziano: {train_dataset.classes[predicted_class]}, Prawdziwy obrazek: {train_dataset.classes[true_class]}')
#     plt.show()


# Uczenie modelu na CPU i GPU
def train_model_on_device(model, device, train_loader, num_epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()  # Czas rozpoczęcia treningu

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    end_time = time.time()  # Czas zakończenia treningu
    training_time = end_time - start_time
    return training_time

# Modele
models = [SimpleCNN()]
model_names = ["SimpleCNN"]

# Porównanie czasu trenowania na CPU i GPU
for model, model_name in zip(models, model_names):
    print(f"Training {model_name}...")

    # Trening na CPU
    cpu_device = torch.device("cpu")
    cpu_model = model
    print(f"Training on CPU...")
    cpu_training_time = train_model_on_device(cpu_model, cpu_device, train_loader)

    # Trening na GPU
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_model = model
    print(f"Training on GPU...")
    gpu_training_time = train_model_on_device(gpu_model, gpu_device, train_loader)

    print(f"Training time on CPU: {cpu_training_time:.2f} seconds")
    print(f"Training time on GPU: {gpu_training_time:.2f} seconds")
    print(f"Speedup: {cpu_training_time / gpu_training_time:.2f}x")
