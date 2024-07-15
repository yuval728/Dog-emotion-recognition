import torch

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted_labels = outputs.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)
        
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_predictions
    
    return train_loss, train_accuracy

def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted_labels = outputs.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
            
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct_predictions / total_predictions
    
    return test_loss, test_accuracy

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.inference_mode():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted_labels = outputs.argmax(dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            
    return predictions
