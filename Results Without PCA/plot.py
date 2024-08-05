import matplotlib.pyplot as plt

# Data
methods = ['Without Alignment', 'Alignment + Input Weight', 'Alignment + Decision Sampling + Input weight']
accuracy = [75, 78.2, 79.5]

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(accuracy)), accuracy, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', width=0.4)

# Labels and Title
plt.xlabel('Kernel Training Method', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Accuracy for Different Kernel Training Methods', fontsize=16)
plt.xticks(range(len(accuracy)), ['Method 1', 'Method 2', 'Method 3'], fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(70, 85)

# Adding the accuracy values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', fontsize=12)

# Adding a legend
plt.legend(bars, methods, fontsize=12, title='Methods', loc='upper left', bbox_to_anchor=(1,1))

plt.tight_layout()
plt.show()
