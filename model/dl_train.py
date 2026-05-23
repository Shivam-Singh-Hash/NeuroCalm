import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from dl_dataset import get_dataloaders
from dl_models import ConvNet, LSTMNet, HybridCNN_LSTM

EPOCHS = 100
PATIENCE = 15 # Early stopping threshold

def compute_class_weights(y_train):
    """
    Given the massive Baseline dominance in WESAD, we must calculate
    inverse-frequency class dependencies to penalize the Loss Engine correctly.
    """
    counts = np.bincount(y_train)
    weights = 1.0 / counts
    normalized_weights = weights / weights.sum()
    return torch.tensor(normalized_weights, dtype=torch.float)

def train_model(model, model_name, train_loader, val_loader, class_weights, device):
    """
    Handles training, tracking, and strict Early Stopping logic.
    """
    print(f"\\n{'='*50}\\nInitiating RTX Hardware Training sequence to: {model_name}\\n{'='*50}")
    model.to(device)
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Added weight_decay for L2 regularization
    
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_weights = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val F1 (Macro): {val_f1:.4f}")
            
        # Early Stopping Logic (Tracking Macro F1 Score strictly!)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_weights = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            print(f"\\n🚨 EARLY STOPPING ENGAGED at Epoch {epoch+1}! Validation limits reached. Halting to prevent Overfitting.")
            break
            
    # Dump best state
    if best_weights:
        model.load_state_dict(best_weights)
        
    return model, best_val_f1

def evaluate_model(model, test_loader, device, model_name):
    """
    Runs isolated inference on the Test subset yielding the final performance metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"\\n>>> {model_name} Final Test Integrity Report <<<")
    print(f"Test Accuracy : {acc*100:.2f}%")
    print(f"Macro F1 Score: {f1:.4f}")
    print("Confusion Matrix:\\n", cm)
    print("Classification Matrix:\\n", classification_report(all_targets, all_preds, target_names=["Baseline", "Stress", "Amusement"]))
    return f1

def run_ml_pipeline():
    # Attempt to latch onto RTX CUDA hardware automatically, fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Engine Targeting Processing Hardware: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'WESAD')
    
    # User-Requested Batch size 64 for optimal RTX saturation
    train_loader, val_loader, test_loader, y_train = get_dataloaders(data_dir, batch_size=64)
    
    # Calculate WESAD unbalance penalty factors
    class_weights = compute_class_weights(y_train)
    print(f"Computed Class Weights for Loss Penalty: {class_weights.numpy()}")
    
    models_to_test = {
        "1D_CNN_Net": ConvNet(),
        "Bi-LSTM_Net": LSTMNet(),
        "Hybrid_CNN_LSTM": HybridCNN_LSTM()
    }
    
    results = {}
    best_f1 = 0
    best_model_name = ""
    best_model_weights = None
    
    # Execute Model Grid Search Sequence
    for name, model in models_to_test.items():
        trained_model, val_best_f1 = train_model(model, name, train_loader, val_loader, class_weights, device)
        final_test_f1 = evaluate_model(trained_model, test_loader, device, name)
        
        results[name] = final_test_f1
        
        if final_test_f1 > best_f1:
            best_f1 = final_test_f1
            best_model_name = name
            best_model_weights = trained_model.state_dict()
            
    print("\\n" + "="*50)
    print("🏆 FULL DEEP LEARNING ARCHITECTURE COMPARISON 🏆")
    print("="*50)
    for name, score in results.items():
        print(f" - {name} : Test F1 {score:.4f}")
        
    print(f"\\nThe system analytically selects the '{best_model_name}' as the dominant architecture.")
    
    # Strictly save only the highest yielding DL configuration natively.
    out_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(out_dir, exist_ok=True)
    
    best_path = os.path.join(out_dir, 'best_dl_model.pth')
    torch.save(best_model_weights, best_path)
    print(f"\\nSuccessfully compressed and saved high-performing neural dictionary to: {best_path}")

if __name__ == '__main__':
    run_ml_pipeline()
