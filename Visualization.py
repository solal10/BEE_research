import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example confusion matrix
confusion_matrix = np.array([[1, 5, 2, 2],
 [5, 2, 2, 6],
 [5, 3, 6, 3],
 [1, 6, 3, 4]])

# Define class labels
class_labels = ['Iteration 1', 'Iteration 2', 'Iteration 7', 'Iteration 8']

# Normalize the confusion matrix
normalized_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Create a heatmap using seaborn
heatmap = sns.heatmap(normalized_matrix, annot=True, fmt=".2f", cmap='Blues', ax=ax)

# Set labels, title, and ticks
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_labels)
ax.yaxis.set_ticklabels(class_labels)

# Rotate the tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Show the plot
plt.tight_layout()
plt.show()
