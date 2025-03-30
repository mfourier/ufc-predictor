from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def validate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return accuracy, f1

