import matplotlib.pyplot as plt
import numpy as np

# Data provided in the table
processors = np.array([8, 16, 24, 32])
nccl_times = np.array([2.88, 2.474515, 2.438371, 2.404416])
scatter_gather_times = np.array([81.1, 43.16, 29.77, 22.367805])

# --- Calculate Speedup ---
# Strong scaling speedup is calculated as Time(1) / Time(P),
# where Time(1) is the execution time on a single processor.
# nccl_speedup = nccl_times[0] / nccl_times
# scatter_gather_speedup = scatter_gather_times[0] / scatter_gather_times

# Ideal linear speedup is equal to the number of processors.
# ideal_speedup = processors

# --- Plotting the Results ---
# Set a plot style and figure size for better aesthetics.
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.family': 'bold'})
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 5))

# Plot each data series with distinct markers and styles.
# plt.plot(processors, ideal_speedup, 'k--', label='Ideal Speedup', marker='s')
plt.plot(processors, scatter_gather_times, '-^', label='Scatter-Gather', markersize=8, linewidth=3)

plt.plot(processors, nccl_times, '-o', label='NCCL All2All (Ours)', markersize=8, linewidth=3)

for i, val in enumerate(nccl_times):
    plt.annotate(f'{val:.2f}', # Format the value to 2 decimal places
                (processors[i], val),
                textcoords="offset points", # Specifies the offset system
                xytext=(0, 10), # Offset the text by 10 points vertically
                ha='center', fontsize=18) # Center the horizontal alignment

# Loop through the Scatter-Gather data points and add text labels
for i, val in enumerate(scatter_gather_times):
    plt.annotate(f'{val:.2f}',
                (processors[i], val),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center', fontsize=18)


# --- Add Titles and Labels ---
plt.title('Time vs. Number of GPUs', fontsize=20)
plt.xlabel('Number of GPUs', fontsize=18)
plt.ylabel('Runtime (s)', fontsize=18)

# Ensure the x-axis ticks correspond to the number of processors.
plt.xticks(processors, fontsize=18)
plt.yticks(fontsize=18)

# Add a legend to identify the lines.
plt.legend(fontsize=18)

# Add a grid and ensure the layout is clean.
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# --- Save the Plot to a File ---
# The plot will be saved in the same directory where the script is run.
plt.savefig('strong_scalability_speedup.pdf')

print("The plot has been successfully saved as strong_scalability_speedup.png")