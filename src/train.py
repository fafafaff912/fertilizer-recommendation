import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm  # PyTorch Image Models library for Vision Transformer

# ------------------------------
# Аргументы командной строки
# ------------------------------
parser = argparse.ArgumentParser(description='Классификация сортов растений')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Путь к корневой папке с данными (должна содержать подпапки train/val/test или все классы в одной папке)')
parser.add_argument('--model', type=str, default='resnet50',
                    choices=['resnet50', 'vit_base_patch16_224'],
                    help='Архитектура модели: resnet50 (CNN) или vit_base_patch16_224 (Vision Transformer)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_classes', type=int, default=10,
                    help='Количество сортов (должно соответствовать данным)')
parser.add_argument('--img_size', type=int, default=224,
                    help='Размер входных изображений (224 для стандартных моделей)')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Папка для сохранения результатов (матрица ошибок, отчёт)')
args = parser.parse_args()

# Создаём выходную директорию, если её нет
os.makedirs(args.output_dir, exist_ok=True)

# ------------------------------
# Подготовка данных и аугментация
# ------------------------------
# Трансформации для тренировочного набора (с аугментацией)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(args.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Трансформации для валидации/теста (без аугментации)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка данных из папок. Ожидается структура:
# data_dir/
#   train/
#       class1/
#       class2/
#       ...
#   val/
#       class1/
#       ...
#   test/
#       class1/
#       ...
# Если нет разделения, можно использовать torchvision.datasets.ImageFolder и разделить вручную,
# но для простоты предполагаем, что данные уже разбиты.
train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transform)
test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=val_transform)

# Сохраняем названия классов для отчёта
class_names = train_dataset.classes
print(f"Найдено классов: {len(class_names)}")
print(f"Классы: {class_names}")

# Загрузчики данных
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# ------------------------------
# Инициализация модели
# ------------------------------
if args.model.startswith('vit'):
    # Vision Transformer из библиотеки timm
    model = timm.create_model(args.model, pretrained=True, num_classes=args.num_classes)
else:
    # ResNet (или другая CNN) из torchvision
    import torchvision.models as models
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Заменяем последний полносвязный слой
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.num_classes)

model = model.to(args.device)

# ------------------------------
# Функция потерь и оптимизатор
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Планировщик темпа обучения (уменьшаем при плато)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

# ------------------------------
# Обучение модели
# ------------------------------
best_val_loss = float('inf')
best_model_path = os.path.join(args.output_dir, 'best_model.pth')

for epoch in range(1, args.epochs + 1):
    # Тренировочная фаза
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]')
    for images, labels in pbar:
        images, labels = images.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': loss.item()})

    train_loss /= len(train_dataset)
    train_acc = correct_train / total_train

    # Валидационная фаза
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(val_dataset)
    val_acc = correct_val / total_val

    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # Сохраняем лучшую модель по валидационной потере
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Лучшая модель сохранена (val_loss={val_loss:.4f})')

    scheduler.step(val_loss)

# ------------------------------
# Тестирование лучшей модели
# ------------------------------
print("\nЗагрузка лучшей модели для тестирования...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_labels = []
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_dataset)
test_acc = correct_test / total_test
print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

# ------------------------------
# Per-class accuracy и confusion matrix
# ------------------------------
# Classification report (per-class precision, recall, f1, accuracy)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("\nClassification Report (per-class metrics):")
print(report)

# Сохраняем отчёт в файл
report_path = os.path.join(args.output_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"Отчёт сохранён в {report_path}")

# Матрица ошибок
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Confusion Matrix')
plt.tight_layout()
cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"Матрица ошибок сохранена в {cm_path}")

# Построим также per-class accuracy (диагональ матрицы, делённая на сумму по строке)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 6))
plt.bar(class_names, per_class_acc)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Per-class Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
per_class_path = os.path.join(args.output_dir, 'per_class_accuracy.png')
plt.savefig(per_class_path)
plt.show()
print(f"График per-class accuracy сохранён в {per_class_path}")
