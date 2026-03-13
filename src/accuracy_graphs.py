import matplotlib.pyplot as plt

# =====================================================
# FINAL FULL-TEST ACCURACY VALUES
# =====================================================
individual_models = {
    "ResNet": 98.81,
    "MobileNet": 88.58,
    "SqueezeNet": 95.65,
    "Custom CNN": 96.06
}

comparison_models = {
    "ResNet": 98.81,
    "MobileNet": 88.58,
    "SqueezeNet": 95.65,
    "Custom CNN": 96.06,
    "Ensemble": 98.81
}

# =====================================================
# GRAPH 1: Individual Models – BAR GRAPH
# =====================================================
plt.figure(figsize=(8,5))
plt.bar(individual_models.keys(), individual_models.values())
plt.ylabel("Accuracy (%)")
plt.title("Individual Model Accuracy (Bar Graph)")
plt.ylim(80, 100)
plt.tight_layout()
plt.savefig("models/individual_accuracy_bar.png", dpi=300)
plt.close()

# =====================================================
# GRAPH 2: Individual Models – LINE GRAPH
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(
    list(individual_models.keys()),
    list(individual_models.values()),
    marker="o"
)
plt.ylabel("Accuracy (%)")
plt.title("Individual Model Accuracy (Line Graph)")
plt.ylim(80, 100)
plt.grid(True)
plt.tight_layout()
plt.savefig("models/individual_accuracy_line.png", dpi=300)
plt.close()

# =====================================================
# GRAPH 3: Comparison – BAR GRAPH (with Ensemble)
# =====================================================
plt.figure(figsize=(8,5))
bars = plt.bar(comparison_models.keys(), comparison_models.values())
bars[-1].set_edgecolor("black")
bars[-1].set_linewidth(2)  # highlight ensemble
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison (Bar Graph)")
plt.ylim(80, 100)
plt.tight_layout()
plt.savefig("models/comparison_accuracy_bar.png", dpi=300)
plt.close()

# =====================================================
# GRAPH 4: Comparison – LINE GRAPH (with Ensemble)
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(
    list(comparison_models.keys()),
    list(comparison_models.values()),
    marker="o"
)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison (Line Graph)")
plt.ylim(80, 100)
plt.grid(True)
plt.tight_layout()
plt.savefig("models/comparison_accuracy_line.png", dpi=300)
plt.close()

print("✅ ALL GRAPHS SAVED SUCCESSFULLY")
print("📁 models/individual_accuracy_bar.png")
print("📁 models/individual_accuracy_line.png")
print("📁 models/comparison_accuracy_bar.png")
print("📁 models/comparison_accuracy_line.png")
