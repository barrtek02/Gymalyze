import pandas as pd
import matplotlib.pyplot as plt

# Load results into a DataFrame
results_df = pd.read_csv(r"/experiments/hyperparameter_tuning_progress.csv")

# Plot Test Accuracy as Scatter Plot
plt.figure(figsize=(10, 6))
for layers in results_df["num_layers"].unique():
    subset = results_df[results_df["num_layers"] == layers]
    plt.scatter(
        subset["hidden_size"],
        subset["test_accuracy"],
        label=f"Layers={layers}",
        alpha=0.7,
    )

plt.title("Test Accuracy vs. Hidden Size by Number of Layers")
plt.xlabel("Hidden Size")
plt.ylabel("Test Accuracy (%)")
plt.legend()
plt.grid()
plt.show()


# Load results into a DataFrame
results_df = pd.read_csv(r"/experiments/hyperparameter_tuning_progress.csv")

# Example: Select the first row's epoch data for plotting
# Adjust the row selection as needed
first_result = results_df.iloc[0]  # Select the first result
epoch_losses = eval(first_result["epoch_losses"])
epoch_accuracies = eval(first_result["epoch_accuracies"])

# Plot Training Loss
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(epoch_losses) + 1), epoch_losses, marker="o", label="Training Loss"
)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(epoch_accuracies) + 1),
    epoch_accuracies,
    marker="o",
    color="orange",
    label="Accuracy",
)
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.legend()
plt.show()
