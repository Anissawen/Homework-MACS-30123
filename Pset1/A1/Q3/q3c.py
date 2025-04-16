import rasterio
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt

# Paths to original Landsat bands
red_band_path = '/project/macs30123/landsat8/LC08_B4.tif'
nir_band_path = '/project/macs30123/landsat8/LC08_B5.tif'

# Read and convert to float32 (for GPU)
with rasterio.open(red_band_path) as red_src:
    red_base = red_src.read(1).astype(np.float32)
with rasterio.open(nir_band_path) as nir_src:
    nir_base = nir_src.read(1).astype(np.float32)

# CUDA kernel definition
mod = SourceModule("""
__global__ void compute_ndvi(float *nir, float *red, float *ndvi, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float denom = nir[idx] + red[idx];
        ndvi[idx] = (denom != 0) ? (nir[idx] - red[idx]) / denom : 0.0f;
    }
}
""", options=["--compiler-options", "-Wall", "-allow-unsupported-compiler"])
compute_ndvi = mod.get_function("compute_ndvi")

block_size = 256

# List of scale factors to simulate larger datasets
scale_factors = [1, 20, 50, 100, 150]

# Print header
print("\n--- NDVI Computation: Serial vs Parallel (Tiled Data) ---")
print(f"{'Tiles':>6} | {'CPU Time (s)':>12} | {'GPU Time (s)':>12} | {'Speedup':>8}")
print("-" * 50)

for scale in scale_factors:
    # Tile the data
    red = np.tile(red_base, scale)
    nir = np.tile(nir_base, scale)

    num_pixels = red.size
    height = red.shape[0]
    width = red.shape[1] if red.ndim > 1 else 1

    # === CPU (Serial) NDVI ===
    start_cpu = time.time()
    ndvi_cpu = (nir - red) / (nir + red)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # === GPU NDVI ===
    ndvi_result = np.empty_like(red)

    # Allocate GPU memory
    red_gpu = cuda.mem_alloc(red.nbytes)
    nir_gpu = cuda.mem_alloc(nir.nbytes)
    ndvi_gpu = cuda.mem_alloc(ndvi_result.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(red_gpu, red)
    cuda.memcpy_htod(nir_gpu, nir)

    # Grid config
    grid_size = (num_pixels + block_size - 1) // block_size

    # Time GPU
    start_gpu = time.time()
    compute_ndvi(nir_gpu, red_gpu, ndvi_gpu, np.int32(num_pixels),
                 block=(block_size, 1, 1), grid=(grid_size, 1))
    cuda.Context.synchronize()
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # Output results
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    print(f"{scale:>6} | {cpu_time:12.4f} | {gpu_time:12.4f} | {speedup:8.2f}")

# Optional: Save one result image
# ndvi_gpu_image = ndvi_result.reshape((height, width))
# plt.imshow(ndvi_gpu_image, cmap='RdYlGn')
# plt.colorbar(label='NDVI')
# plt.title(f"NDVI GPU {scale}x")
# plt.savefig(f"ndvi_gpu_{scale}x.png")
