import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, train_loader, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def finetune(self, image, label):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(image.view(-1, 28 * 28))
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        print(f"Finetune Loss: {loss.item():.4f}")
