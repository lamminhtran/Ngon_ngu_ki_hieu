import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, log_loss, accuracy_score

if __name__ == "__main__":
    # Get the pose name from argument
    parser = argparse.ArgumentParser("Training model")

    # Add and parse the arguments
    parser.add_argument("--model_name", help="Name of the model",
                        type=str, default="model")
    parser.add_argument("--dir", help="Location of the model",
                        type=str, default="models")
    args = parser.parse_args()

    # Train X, y and mapping
    X, y, mapping = [], [], dict()

    # Read in the data from data folder
    class_counts = {}  # To store the number of images per class
    for current_class_index, pose_file in enumerate(os.scandir("data")):
        # Load pose data
        file_path = f"data/{pose_file.name}"
        pose_data = np.load(file_path)

        # Add to training data
        X.append(pose_data)
        y += [current_class_index] * pose_data.shape[0]

        # Add to mapping
        mapping[current_class_index] = pose_file.name.split(".")[0]

        # Store class count
        class_counts[mapping[current_class_index]] = pose_data.shape[0]

    # Convert to Numpy
    X, y = np.vstack(X), np.array(y)

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Display data distribution
    print("\nData distribution:")
    for pose, count in class_counts.items():
        print(f"- {pose}: {count} images")
    print(f"\nTrain set size: {len(y_train)} images")
    print(f"Test set size: {len(y_test)} images\n")

    # Train models
    models = {
        "SVM": SVC(decision_function_shape='ovo', kernel='rbf'),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, learning_rate="constant",
                                             eta0=0.01),
    }

    results = {}
    log_reg_losses, log_reg_accuracies = [], []

    for name, model in models.items():
        # Train Logistic Regression manually to track loss and accuracy
        if name == "Logistic Regression":
            for epoch in range(50):  # Training for 50 epochs
                model.partial_fit(X_train, y_train, classes=np.unique(y))

                # Calculate loss
                probs = model.predict_proba(X_train)
                loss = log_loss(y_train, probs)
                log_reg_losses.append(loss)

                # Calculate accuracy
                y_train_pred = model.predict(X_train)
                acc = accuracy_score(y_train, y_train_pred)
                log_reg_accuracies.append(acc)
        else:
            # Train model
            model.fit(X_train, y_train)

        # Evaluate model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Store results
        results[name] = {
            "model": model,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "y_pred": y_pred,
        }

        # Save the model
        os.makedirs(args.dir, exist_ok=True)
        model_path = os.path.join(args.dir, f"{args.model_name}_{name.lower().replace(' ', '_')}.pkl")
        with open(model_path, "wb") as file:
            pickle.dump((model, mapping), file)
        print(f"Saved {name} model to {model_path}")

    # Display results
    for name, result in results.items():
        print(
            f"{name} -> Train accuracy: {round(result['train_accuracy'] * 100, 2)}% - Test accuracy: {round(result['test_accuracy'] * 100, 2)}%")

    # Visualization
    # Bar plot for accuracy
    labels = list(results.keys())
    train_accuracies = [result["train_accuracy"] for result in results.values()]
    test_accuracies = [result["test_accuracy"] for result in results.values()]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, train_accuracies, width, label='Train Accuracy')
    plt.bar(x + width / 2, test_accuracies, width, label='Test Accuracy')
    plt.ylabel('Accuracy')
    plt.title('Model Train and Test Accuracy Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

    # Confusion matrices
    for name, result in results.items():
        cm = confusion_matrix(y_test, result["y_pred"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(mapping.values()),
                    yticklabels=list(mapping.values()))
        plt.title(f"Confusion Matrix for {name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    # Logistic Regression Loss and Accuracy Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(log_reg_losses, label="Log Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Logistic Regression Loss over Epochs")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(log_reg_accuracies, label="Train Accuracy", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Logistic Regression Accuracy over Epochs")
    plt.legend()
    plt.show()

    # Classification reports
    for name, result in results.items():
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, result["y_pred"], target_names=list(mapping.values())))
