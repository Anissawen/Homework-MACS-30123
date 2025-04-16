import matplotlib.pyplot as plt

cores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Corresponding total elapsed times (in seconds) for each run
elapsed_times = [
    0.0236, 0.0137, 0.0096, 0.0059, 0.0048,
    0.0044, 0.0038, 0.0034, 0.0031, 0.0027,
    0.0026, 0.0024, 0.0023, 0.0021, 0.0021,
    0.0019, 0.0019, 0.0019, 0.0018, 0.0016
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(cores, elapsed_times, marker='o', linestyle='-', color='b')

# Labeling axes and title
plt.xlabel("Number of MPI Processes (Cores)")
plt.ylabel("Total Elapsed Time (seconds)")
plt.title("Scaling Study: Computation Time vs. Number of MPI Processes")
plt.xticks(cores)  # Show integer ticks for each core count
plt.grid(True)

# Save the plot to the current directory as a PNG file
plt.savefig("q1b_plot.png", dpi=300, bbox_inches="tight")