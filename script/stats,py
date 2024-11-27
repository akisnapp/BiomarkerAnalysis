import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def precision(outputs, correct_labels):
    _, predicted = outputs.max(1)  # Get predicted class (max probability)
    precision = precision_score(correct_labels.cpu(), predicted.cpu(), average='weighted', zero_division=1)
    return precision

def recall(outputs, correct_labels):
    _, predicted = outputs.max(1)  # Get predicted class (max probability)
    recall = recall_score(correct_labels.cpu(), predicted.cpu(), average='weighted', zero_division=1)
    return recall

def compute_f1_score(outputs, correct_labels):
    _, predicted = outputs.max(1)  # Get the predicted class (max probability)
    f1 = f1_score(correct_labels.cpu(), predicted.cpu(), average='weighted', zero_division=1)
    return f1

def compute_specificity(outputs, correct_labels):
    _, predicted = outputs.max(1)  # Get the predicted class (max probability)
    
    tn, fp, fn, tp = confusion_matrix(correct_labels.cpu(), predicted.cpu()).ravel()

    if tn + fp == 0:
        return 0

    specificity = tn / (tn + fp)
    return specificity


def plot_performance(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    # Training loss + validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="red")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Training accuracy + validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color="blue")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="red")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    pass #TODO: Fill this out once everything is finished so that inputs to statistics functions are known


if __name__ == "__main__":
    main()
