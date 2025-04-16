import rasterio
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time


# Load the Satellite Bands
red_band_path = '/project/macs30123/landsat8/LC08_B4.tif'  # red
nir_band_path = '/project/macs30123/landsat8/LC08_B5.tif'   # nir

# Read bands into numpy arrays
with rasterio.open(red_band_path) as red_src:
    red = red_src.read(1).astype(np.float64)

with rasterio.open(nir_band_path) as nir_src:
    nir = nir_src.read(1).astype(np.float64)

# === CPU NDVI (Serial) Calculation ===
start_cpu = time.time()
ndvi_cpu = (nir - red) / (nir + red)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# === GPU NDVI (CUDA with float32) ===
# Convert to float32 for GPU compatibility
red_gpu_f32 = red.astype(np.float32)
nir_gpu_f32 = nir.astype(np.float32)

height, width = red.shape
num_pixels = height * width

# Flatten for GPU
red_flat = red_gpu_f32.ravel()
nir_flat = nir_gpu_f32.ravel()
ndvi_result = np.empty_like(red_flat)

# Allocate GPU memory
red_gpu = cuda.mem_alloc(red_flat.nbytes)
nir_gpu = cuda.mem_alloc(nir_flat.nbytes)
ndvi_gpu = cuda.mem_alloc(ndvi_result.nbytes)

# Copy to device
cuda.memcpy_htod(red_gpu, red_flat)
cuda.memcpy_htod(nir_gpu, nir_flat)

# CUDA Kernel
mod = SourceModule("""
__global__ void compute_ndvi(float *nir, float *red, float *ndvi, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float denom = nir[idx] + red[idx];
        if (denom != 0) {
            ndvi[idx] = (nir[idx] - red[idx]) / denom;
        } else {
            ndvi[idx] = 0;
        }
    }
}
""", options=["--compiler-options", "-Wall", "-allow-unsupported-compiler"])

compute_ndvi = mod.get_function("compute_ndvi")
block_size = 256
grid_size = (num_pixels + block_size - 1) // block_size

# Time GPU NDVI
start_gpu = time.time()
compute_ndvi(nir_gpu, red_gpu, ndvi_gpu, np.int32(num_pixels),
             block=(block_size, 1, 1), grid=(grid_size, 1))
cuda.Context.synchronize()  # Make sure kernel finishes
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

# Copy back to CPU
cuda.memcpy_dtoh(ndvi_result, ndvi_gpu)
ndvi_gpu_image = ndvi_result.reshape((height, width))

# === Save the NDVI Image ===
plt.imshow(ndvi_gpu_image, cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.title("NDVI from GPU")
plt.savefig("gpu_ndvi.png")
plt.close()

# === Print Performance Comparison ===
print("\n--- NDVI Computation Performance ---")
print(f"✅ GPU (CUDA) Time     : {gpu_time:.4f} seconds")
print(f"⏱ CPU (Serial NumPy)  : {cpu_time:.4f} seconds")
print(f"⚡ Speedup (CPU / GPU) : {cpu_time / gpu_time:.2f}x")