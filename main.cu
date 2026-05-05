#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

// Silence "set but never used" warnings from stb_image.h only.
#pragma nv_diag_suppress 550
#include "stb_image.h"
#include "stb_image_write.h"
#pragma nv_diag_default 550
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <chrono>
#include <cuda_runtime.h>

#define NUM_CHANNELS 1

// Abort on any CUDA error.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)


// ---------- CPU baselines ----------

// Single pass over the image, tracking running min and max.
void min_max_of_img_host(uint8_t* img, uint8_t* min, uint8_t* max, int width, int height) {
    int max_tmp = 0;
    int min_tmp = 255;
    for (int n = 0; n < width * height; n++) {
        max_tmp = (img[n] > max_tmp) ? img[n] : max_tmp;
        min_tmp = (img[n] < min_tmp) ? img[n] : min_tmp;
    }
    *max = max_tmp;
    *min = min_tmp;
}

// Subtract the same value from every pixel (uint8_t wraparound is fine here -
// callers pass sub_value <= every pixel).
void sub_host(uint8_t* img, uint8_t sub_value, int width, int height) {
    for (int n = 0; n < width * height; n++) {
        img[n] -= sub_value;
    }
}

// Multiply every pixel by `constant`. Float result is implicitly truncated
// back to uint8_t, GPU mirrors this exactly so outputs match byte-for-byte.
void scale_host(uint8_t* img, float constant, int width, int height) {
    for (int n = 0; n < width * height; n++) {
        img[n] = img[n] * constant;
    }
}


// ---------- GPU kernels ----------

// KERNEL 2: subtract sub_value from every pixel.
__global__ void sub_kernel(uint8_t* img, uint8_t sub_value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        img[idx] -= sub_value;
    }
}

// KERNEL 3: scale every pixel by constant (float -> uint8_t truncation).
__global__ void scale_kernel(uint8_t* img, float constant, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        img[idx] = img[idx] * constant;
    }
}

// KERNEL 1 (naive): each thread updates a single global min/max with atomics.
// atomicMin/atomicMax need int*, so caller passes ints initialized to 255/0.
__global__ void minmax_atomic_kernel(const uint8_t* img, int n,
                                     int* g_min, int* g_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int v = img[idx];
        atomicMin(g_min, v);
        atomicMax(g_max, v);
    }
}

// KERNEL 1 (reduction): per-block tree reduction in shared memory, then one
// atomicMin/atomicMax per block. Dynamic shared memory holds two parallel
// arrays (s_min, s_max) of size blockDim.
__global__ void minmax_reduce_kernel(const uint8_t* img, int n,
                                     int* g_min, int* g_max) {
    extern __shared__ int s_data[];
    int* s_min = s_data;
    int* s_max = s_data + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Out-of-range threads load identity values so the reduction loop has no bounds checks.
    s_min[tid] = (idx < n) ? (int)img[idx] : 255;
    s_max[tid] = (idx < n) ? (int)img[idx] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other_min = s_min[tid + stride];
            int other_max = s_max[tid + stride];
            if (other_min < s_min[tid]) s_min[tid] = other_min;
            if (other_max > s_max[tid]) s_max[tid] = other_max;
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(g_min, s_min[0]);
        atomicMax(g_max, s_max[0]);
    }
}


// ---------- Benchmark pipelines ----------
// CPU timing uses std::chrono. GPU timing uses cudaEvents and reports two
// intervals: kernels-only, and total (includes H2D + D2H transfers).

struct CpuResult { float ms; };
struct GpuResult { float total_ms; float kernels_ms; uint8_t min; uint8_t max; };

// Selects which KERNEL 1 implementation the GPU pipeline uses.
enum MinMaxKind { MINMAX_ATOMIC = 0, MINMAX_REDUCE = 1 };

// Run the full CPU pipeline (min/max -> subtract -> scale) on `out` in place.
static CpuResult run_cpu_pipeline(uint8_t* out, int width, int height) {
    using clock = std::chrono::high_resolution_clock;
    uint8_t mn, mx;

    auto t0 = clock::now();
    min_max_of_img_host(out, &mn, &mx, width, height);
    float k = 255.0f / (mx - mn);
    sub_host(out, mn, width, height);
    scale_host(out, k, width, height);
    auto t1 = clock::now();

    CpuResult r;
    r.ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return r;
}

// Run the full GPU pipeline on `out` in place. Returns kernels-only time
// (GPU work) and total time (work + H2D/D2H copies), plus the min/max found.
static GpuResult run_gpu_pipeline(uint8_t* out, int width, int height,
                                  MinMaxKind mm) {
    const int n_pixels = width * height;
    const size_t img_bytes = (size_t)n_pixels;

    // Device buffers: one for the image, two scalars for min/max.
    uint8_t* d_image = nullptr;
    int* d_min = nullptr;
    int* d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_image, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));

    cudaEvent_t e_total_begin, e_kernels_begin, e_kernels_end, e_total_end;
    CUDA_CHECK(cudaEventCreate(&e_total_begin));
    CUDA_CHECK(cudaEventCreate(&e_kernels_begin));
    CUDA_CHECK(cudaEventCreate(&e_kernels_end));
    CUDA_CHECK(cudaEventCreate(&e_total_end));

    // Launch config: 256 threads/block, enough blocks to cover every pixel,
    // shared memory sized for two int arrays (min + max) of `block_size`.
    const int block_size = 256;
    const int grid_size  = (n_pixels + block_size - 1) / block_size;
    const size_t shmem_bytes = 2 * block_size * sizeof(int);

    CUDA_CHECK(cudaEventRecord(e_total_begin));

    // H2D: upload image and seed min/max with identity values.
    CUDA_CHECK(cudaMemcpy(d_image, out, img_bytes, cudaMemcpyHostToDevice));
    int init_min = 255, init_max = 0;
    CUDA_CHECK(cudaMemcpy(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(e_kernels_begin));

    // KERNEL 1 - pick atomic or reduction variant.
    if (mm == MINMAX_REDUCE) {
        minmax_reduce_kernel<<<grid_size, block_size, shmem_bytes>>>(
            d_image, n_pixels, d_min, d_max);
    } else {
        minmax_atomic_kernel<<<grid_size, block_size>>>(
            d_image, n_pixels, d_min, d_max);
    }
    CUDA_CHECK(cudaGetLastError());

    // Pull min/max back so the host can compute the scale factor.
    int min_int, max_int;
    CUDA_CHECK(cudaMemcpy(&min_int, d_min, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&max_int, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    uint8_t mn = (uint8_t)min_int;
    uint8_t mx = (uint8_t)max_int;
    float scale_constant = 255.0f / (mx - mn);

    // KERNEL 2 then KERNEL 3.
    sub_kernel<<<grid_size, block_size>>>(d_image, mn, n_pixels);
    CUDA_CHECK(cudaGetLastError());
    scale_kernel<<<grid_size, block_size>>>(d_image, scale_constant, n_pixels);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(e_kernels_end));

    // D2H: bring the processed pixels back to the host buffer.
    CUDA_CHECK(cudaMemcpy(out, d_image, img_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(e_total_end));
    CUDA_CHECK(cudaEventSynchronize(e_total_end));

    GpuResult r;
    CUDA_CHECK(cudaEventElapsedTime(&r.total_ms,   e_total_begin,   e_total_end));
    CUDA_CHECK(cudaEventElapsedTime(&r.kernels_ms, e_kernels_begin, e_kernels_end));
    r.min = mn;
    r.max = mx;

    CUDA_CHECK(cudaEventDestroy(e_total_begin));
    CUDA_CHECK(cudaEventDestroy(e_kernels_begin));
    CUDA_CHECK(cudaEventDestroy(e_kernels_end));
    CUDA_CHECK(cudaEventDestroy(e_total_end));
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_min));
    CUDA_CHECK(cudaFree(d_max));
    return r;
}


int main() {
    // Test images at increasing resolutions.
    const char* samples[] = {
        "./samples/640x426.bmp",
        "./samples/1280x843.bmp",
        "./samples/1920x1280.bmp",
        "./samples/5184x3456.bmp",
    };
    const int n_samples = sizeof(samples) / sizeof(samples[0]);

    // One warmup (drops JIT/context-init cost) plus N timed iterations averaged.
    const int n_warmup = 1;
    const int n_iters  = 5;

    // Table header.
    printf("\n%-22s | %-8s | %10s | %12s | %12s | %10s | %10s | %s\n",
           "image", "minmax", "CPU ms", "GPU total ms", "GPU kern ms",
           "spd total", "spd kern", "match?");
    printf("-----------------------+----------+------------+--------------+--------------+------------+------------+--------\n");

    for (int s = 0; s < n_samples; s++) {
        int width, height, bpp;
        uint8_t* original = stbi_load(samples[s], &width, &height, &bpp, NUM_CHANNELS);
        if (!original) {
            fprintf(stderr, "Failed to load %s\n", samples[s]);
            continue;
        }
        const size_t img_bytes = (size_t)width * height;

        // Two scratch buffers so CPU and GPU don't clobber each other's input.
        uint8_t* buf_cpu = (uint8_t*)malloc(img_bytes);
        uint8_t* buf_gpu = (uint8_t*)malloc(img_bytes);

        // Run both KERNEL 1 variants for comparison.
        const MinMaxKind variants[2]  = { MINMAX_ATOMIC, MINMAX_REDUCE };
        const char*      var_label[2] = { "atomic",      "reduce"      };

        // CPU time is independent of the variant, so pool every iteration.
        float cpu_sum_all = 0.0f;
        int cpu_count_all = 0;

        // Per-variant accumulators.
        float gpu_total[2] = {0.0f, 0.0f};
        float gpu_kern[2]  = {0.0f, 0.0f};
        bool  match[2]     = {false, false};
        uint8_t mn_v[2] = {0,0}, mx_v[2] = {0,0};

        for (int v = 0; v < 2; v++) {
            // Warmup runs are discarded (first CUDA call pays JIT/init cost).
            for (int i = 0; i < n_warmup; i++) {
                memcpy(buf_cpu, original, img_bytes);
                run_cpu_pipeline(buf_cpu, width, height);
                memcpy(buf_gpu, original, img_bytes);
                run_gpu_pipeline(buf_gpu, width, height, variants[v]);
            }
            // Timed runs.
            for (int i = 0; i < n_iters; i++) {
                memcpy(buf_cpu, original, img_bytes);
                CpuResult c = run_cpu_pipeline(buf_cpu, width, height);
                cpu_sum_all += c.ms;
                cpu_count_all++;

                memcpy(buf_gpu, original, img_bytes);
                GpuResult g = run_gpu_pipeline(buf_gpu, width, height, variants[v]);
                gpu_total[v] += g.total_ms;
                gpu_kern[v]  += g.kernels_ms;
                mn_v[v] = g.min;
                mx_v[v] = g.max;
            }
            gpu_total[v] /= n_iters;
            gpu_kern[v]  /= n_iters;

            // Correctness check, GPU output must match CPU output byte-for-byte.
            match[v] = (memcmp(buf_cpu, buf_gpu, img_bytes) == 0);

            // Save one CPU + one GPU output per image into output_cpu/ and output_gpu/,
            // reusing the source filename so inputs and outputs line up by name.
            if (variants[v] == MINMAX_REDUCE) {
                mkdir("./output_cpu", 0755);
                mkdir("./output_gpu", 0755);
                const char* slash = strrchr(samples[s], '/');
                const char* base = slash ? slash + 1 : samples[s];
                char path_gpu[128], path_cpu[128];
                snprintf(path_gpu, sizeof(path_gpu), "./output_gpu/%s", base);
                snprintf(path_cpu, sizeof(path_cpu), "./output_cpu/%s", base);
                stbi_write_bmp(path_gpu, width, height, 1, buf_gpu);
                stbi_write_bmp(path_cpu, width, height, 1, buf_cpu);
            }
        }
        const float cpu_ms = cpu_sum_all / cpu_count_all;

        // Print one row per variant, then the min/max found for this image.
        for (int v = 0; v < 2; v++) {
            char tag[32];
            snprintf(tag, sizeof(tag), "%dx%d", width, height);
            printf("%-22s | %-8s | %10.3f | %12.3f | %12.3f | %9.2fx | %9.2fx | %s\n",
                   v == 0 ? tag : "",
                   var_label[v],
                   cpu_ms,
                   gpu_total[v], gpu_kern[v],
                   cpu_ms / gpu_total[v], cpu_ms / gpu_kern[v],
                   match[v] ? "ok" : "MISMATCH");
        }
        printf("%-22s | min=%u max=%u (used by all variants)\n",
               "", (unsigned)mn_v[1], (unsigned)mx_v[1]);

        free(buf_cpu);
        free(buf_gpu);
        stbi_image_free(original);
    }

    printf("\nWrote ./output_cpu/<WxH>.bmp and ./output_gpu/<WxH>.bmp for each sample.\n");
    printf("Verify byte-identical with e.g.:  cmp output_cpu/640x426.bmp output_gpu/640x426.bmp\n");
    return 0;
}
