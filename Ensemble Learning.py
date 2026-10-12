
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay
)

# 1. Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Define base models
tree = DecisionTreeClassifier(random_state=42)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

# 4. Define ensemble models
models = {
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        max_samples=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "Voting": VotingClassifier(
        estimators=[
            ("tree", tree),
            ("knn", knn),
            ("lr", lr)
        ],
        voting="hard"
    ),

    "Stacking": StackingClassifier(
        estimators=[
            ("tree", tree),
            ("knn", knn),
            ("lr", lr)
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5
    ),

    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
}

# 5. Train and evaluate models
results = {}
predictions_dict = {}

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy
    predictions_dict[name] = predictions

    print(f"{name} Accuracy: {accuracy:.4f}")

# 6. Bar chart: compare accuracy
names = list(results.keys())
accuracies = [results[name] * 100 for name in names]

plt.figure(figsize=(9, 5))
bars = plt.bar(names, accuracies)

plt.title("Ensemble Models - Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()

# 7. Bar chart: correct vs incorrect predictions
correct = [
    np.sum(predictions_dict[name] == y_test)
    for name in names
]

incorrect = [
    len(y_test) - correct[i]
    for i in range(len(names))
]

plt.figure(figsize=(9, 5))
x = np.arange(len(names))
width = 0.35

plt.bar(x - width / 2, correct, width, label="Correct")
plt.bar(x + width / 2, incorrect, width, label="Incorrect")

plt.xticks(x, names)
plt.title("Correct vs Incorrect Predictions")
plt.xlabel("Model")
plt.ylabel("Number of Images")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# 8. Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, name in zip(axes.flat, names):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions_dict[name],
        ax=ax,
        colorbar=False
    )
    ax.set_title(name)

plt.suptitle("Confusion Matrices", fontsize=16)
plt.tight_layout()
plt.show()

# 9. Print final ranking
print("\nFinal Ranking:")

for rank, (name, accuracy) in enumerate(
    sorted(results.items(), key=lambda item: item[1], reverse=True),
    start=1
):
    print(f"{rank}. {name}: {accuracy * 100:.2f}%")