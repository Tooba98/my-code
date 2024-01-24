import matplotlib.pyplot as plt

# Given values
precision = 0.802
recall = 0.620
f1_score = 0.705

# Plot the F1-score line graph
plt.figure(figsize=(8, 6))
plt.plot([f1_score], marker='o', linestyle='-', color='b', label='F1-score')

# Annotate point with precision and recall values
plt.annotate(f'Precision: {precision:.3f}\nRecall: {recall:.3f}', (0, f1_score), textcoords="offset points", xytext=(0,5), ha='center')

plt.xticks([0], ['Point'])
plt.xlabel('Points')
plt.ylabel('F1-score')
plt.title('F1-score for Given Precision, Recall, and F1-score Values')
plt.legend()
plt.grid(True)
plt.show()
