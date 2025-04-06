# /utils/helpers.py

def is_pytorch_model(model):
    return isinstance(model, torch.nn.Module)

def get_predictions(model, X_test):
    """
    Returns model predictions and probabilities if applicable.
    """
    if is_pytorch_model(model):
        model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_tensor).view(-1)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)
        return preds, probs
    else:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.decision_function(X_test)
            # Apply sigmoid for SVMs or linear models
            probs = 1 / (1 + np.exp(-probs))
        preds = model.predict(X_test)
        return preds, probs