import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example confusion matrix
confusion_matrix = np.array([[1, 1, 4, 0, 0, 1, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 2, 0, 1, 3, 1, 0, 0, 0],
 [1, 1, 1, 2, 2, 0, 0, 0, 2, 0, 0, 0],
 [1, 0, 1, 0, 1, 2, 0, 1, 2, 0, 0, 0],
 [1, 1, 1, 3, 3, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 3, 2, 1, 0, 0, 1, 2, 0, 0, 0],
 [0, 0, 1, 1, 0, 0, 3, 0, 2, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 1, 2, 3, 0, 0, 0],
 [1, 0, 2, 2, 2, 0 ,1, 1, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 4],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 4]])

# Define class labels
class_labels = ['Hive 2','Hive 5','Hive 8', 'Hive 1', 'Hive 4', 'Hive 7', 'Hive 3', 'Hive 6', 'Hive 9', 'Hive 11', 'Hive 10', 'Hive 12']

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
