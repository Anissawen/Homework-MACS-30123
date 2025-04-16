import matplotlib.pyplot as plt

cores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Corresponding total elapsed times (in seconds) for each run
elapsed_times = [
    0.0282, 0.0154, 0.0114, 0.0081, 0.0064,
    0.0058, 0.0055, 0.0036, 0.0035, 0.0025,
    0.0026, 0.0024, 0.0022, 0.0020, 0.0020,
    0.0018, 0.0017, 0.0016, 0.0015, 0.0012
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