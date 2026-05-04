// Do not alter the preprocessor directives
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

#define NUM_CHANNELS 1

// Tiny helper, every CUDA call returns a cudaError_t. If it isn't cudaSuccess,
// something went wrong (out of memory, bad pointer, kernel launch failure, etc.).
// Since I am using Google Colab, 
// this macro prints the error and aborts so we don't keep running on a broken state or session.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)


// ---------- CPU baselines (unchanged from main.cpp, kept for reference) ----------

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

void sub_host(uint8_t* img, uint8_t sub_value, int width, int height) {
    for (int n = 0; n < width * height; n++) {
        img[n] -= sub_value;
    }
}

void scale_host(uint8_t* img, float constant, int width, int height) {
    for (int n = 0; n < width * height; n++) {
        img[n] = img[n] * constant; // implicit float -> uint8_t truncation
    }
}


// ---------- Step 2: KERNEL 2 — subtract nMin from every pixel ----------
// __global__ means: this function is launched from the host (CPU) but runs on
// the device (GPU). Every thread that the launch creates executes this body
// independently and concurrently.
//
// The launch will create (grid * block) threads in total. Each thread figures
// out which pixel it owns by computing a unique linear index from its position
// in the grid:
//
//     idx = blockIdx.x * blockDim.x + threadIdx.x
//            ^^^^^^^^^   ^^^^^^^^^^   ^^^^^^^^^^^
//            which block | block size | which thread inside the block
//
// Because we round the grid size up, we may launch a few more threads than we
// have pixels — those extra threads just exit via the bounds check.
__global__ void sub_kernel(uint8_t* img, uint8_t sub_value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        img[idx] -= sub_value;   // same uint8_t arithmetic as sub_host
    }
}


// ---------- Step 3: KERNEL 3 — scale every pixel by 255 / (nMax - nMin) ----------
// Same one-thread-per-pixel pattern as sub_kernel. The only differences:
//   1) the operation is a multiply by a float `constant`,
//   2) the result is stored back as uint8_t, so the float result is implicitly
//      truncated (not rounded) — exactly what scale_host does on the CPU. We
//      keep the same behavior so the GPU output matches the CPU output bit-for-bit.
__global__ void scale_kernel(uint8_t* img, float constant, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        img[idx] = img[idx] * constant;   // float -> uint8_t implicit truncation
    }
}


// ---------- Step 4a: KERNEL 1 — naive min/max with global atomics ----------
// Every thread reads one pixel and atomically updates a single global min and
// a single global max. This is correct but SLOW: when many threads race on the
// same address, the hardware must serialize their updates. Useful as a
// baseline — and as a teaching example of *why* a real reduction matters.
//
// Note: CUDA's atomicMin/atomicMax don't take uint8_t. We promote pixel values
// to int, and the caller initializes g_min=255, g_max=0 before launching.
__global__ void minmax_atomic_kernel(const uint8_t* img, int n,
                                     int* g_min, int* g_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int v = img[idx];
        atomicMin(g_min, v);
        atomicMax(g_max, v);
    }
}


// ---------- Step 4b: KERNEL 1 — proper parallel reduction in shared memory ----------
// This is the version the assignment is really asking for (see the lecture's
// "parallel reduction" slides). Big idea:
//
//   1) Each block loads its tile of pixels into __shared__ memory. Shared
//      memory lives on-chip per SM — about ~100x lower latency than global
//      DRAM, so repeated reads/writes during the reduction are fast.
//
//   2) Inside the block we do a TREE REDUCTION. Start with `stride = blockDim/2`.
//      Threads with tid < stride combine s[tid] with s[tid + stride]. Halve
//      the stride and repeat until stride == 0. After log2(blockDim) steps,
//      s[0] holds the block's min (and a parallel array holds the block's max).
//
//        block_size = 8 example, reducing min:
//          [ 5  3  9  1  7  4  6  2 ]   stride=4 -> compare i with i+4
//          [ 5  3  6  1  ]               stride=2
//          [ 5  1 ]                      stride=1
//          [ 1 ]                         done
//
//   3) Thread 0 of each block writes its block-min/max to the global answer
//      via atomicMin/atomicMax. Now atomics contend at most once per block
//      (instead of once per pixel), which is orders of magnitude less.
//
// __syncthreads() is essential between steps: it forces every thread in the
// block to reach this point before any moves on. Without it, some threads
// would read s[tid+stride] before others had written it -> race condition,
// wrong answer.
//
// Shared-memory layout: we need TWO arrays (one for min, one for max). We
// use dynamic shared memory (size set at launch time via the third <<<>>>
// argument) so we can size it to blockDim. Pointers s_min and s_max carve
// the single buffer into two halves.
__global__ void minmax_reduce_kernel(const uint8_t* img, int n,
                                     int* g_min, int* g_max) {
    extern __shared__ int s_data[];
    int* s_min = s_data;
    int* s_max = s_data + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load. Out-of-range threads load IDENTITY values: 255 for min (won't
    // win), 0 for max (won't win). This way we don't need bounds checks
    // inside the reduction loop.
    s_min[tid] = (idx < n) ? (int)img[idx] : 255;
    s_max[tid] = (idx < n) ? (int)img[idx] : 0;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other_min = s_min[tid + stride];
            int other_max = s_max[tid + stride];
            if (other_min < s_min[tid]) s_min[tid] = other_min;
            if (other_max > s_max[tid]) s_max[tid] = other_max;
        }
        __syncthreads();
    }

    // One atomic per block, not per pixel — almost no contention.
    if (tid == 0) {
        atomicMin(g_min, s_min[0]);
        atomicMax(g_max, s_max[0]);
    }
}


// ---------- Step 5: timing helpers + benchmark pipeline ----------
//
// We compare the CPU pipeline against the GPU pipeline. To make the comparison
// fair both must start from the same input pixels. They each operate on their
// own copy of the buffer (`out` parameter).
//
// CPU timing: std::chrono::high_resolution_clock — wall-clock time around the
// three host functions called in sequence.
//
// GPU timing: we use cudaEvents (recorded ON the GPU stream). Two separate
// intervals are measured:
//   - "kernels only": from just before the first kernel to just after the last
//     kernel. This is the work the GPU literally does.
//   - "total"       : also includes the H2D and D2H pixel copies. This is what
//     the user actually waits for in practice.
//
// Why measure both? On small images the PCIe copies can dominate the kernel
// time and the GPU may even be slower than the CPU end-to-end. On big images
// the kernels dominate and the GPU pulls ahead by a wide margin. Showing both
// numbers makes that crossover visible in your report.

struct CpuResult { float ms; };
struct GpuResult { float total_ms; float kernels_ms; uint8_t min; uint8_t max; };

// Which min/max kernel to use. The other two kernels (subtract, scale) are
// the same in both modes — only KERNEL 1 differs. We expose this so the bench
// can directly compare the naive atomic baseline against the proper reduction.
enum MinMaxKind { MINMAX_ATOMIC = 0, MINMAX_REDUCE = 1 };

// Run the CPU baseline pipeline on `out` (modified in place). Returns elapsed ms.
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

// Run the GPU pipeline on `out` (modified in place). Returns elapsed times
// plus the min/max the GPU computed so we can sanity-check against the CPU.
// `mm` selects which KERNEL 1 implementation to use.
static GpuResult run_gpu_pipeline(uint8_t* out, int width, int height,
                                  MinMaxKind mm) {
    const int n_pixels = width * height;
    const size_t img_bytes = (size_t)n_pixels;

    // Device allocations.
    uint8_t* d_image = nullptr;
    int* d_min = nullptr;
    int* d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_image, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));

    // Events for timing. Unlike chrono, cudaEvents are recorded into the GPU
    // stream itself — the GPU stamps them between the operations actually
    // executing on the device, so they don't include CPU-side bookkeeping.
    cudaEvent_t e_total_begin, e_kernels_begin, e_kernels_end, e_total_end;
    CUDA_CHECK(cudaEventCreate(&e_total_begin));
    CUDA_CHECK(cudaEventCreate(&e_kernels_begin));
    CUDA_CHECK(cudaEventCreate(&e_kernels_end));
    CUDA_CHECK(cudaEventCreate(&e_total_end));

    const int block_size = 256;
    const int grid_size  = (n_pixels + block_size - 1) / block_size;
    const size_t shmem_bytes = 2 * block_size * sizeof(int);

    // ---- begin total interval ----
    CUDA_CHECK(cudaEventRecord(e_total_begin));

    // H2D: image + initialize d_min/d_max identity values.
    CUDA_CHECK(cudaMemcpy(d_image, out, img_bytes, cudaMemcpyHostToDevice));
    int init_min = 255, init_max = 0;
    CUDA_CHECK(cudaMemcpy(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice));

    // ---- begin kernels-only interval ----
    CUDA_CHECK(cudaEventRecord(e_kernels_begin));

    // Kernel 1: select naive atomic vs shared-memory reduction.
    if (mm == MINMAX_REDUCE) {
        minmax_reduce_kernel<<<grid_size, block_size, shmem_bytes>>>(
            d_image, n_pixels, d_min, d_max);
    } else {
        minmax_atomic_kernel<<<grid_size, block_size>>>(
            d_image, n_pixels, d_min, d_max);
    }
    CUDA_CHECK(cudaGetLastError());

    // Pull the two scalars back so the host can compute scale_constant. This
    // small DtoH copy belongs to the kernels-only interval because it's part
    // of the GPU pipeline's critical path. (cudaMemcpy of <= 64KB on the
    // default stream synchronizes — it implicitly waits for kernel 1.)
    int min_int, max_int;
    CUDA_CHECK(cudaMemcpy(&min_int, d_min, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&max_int, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    uint8_t mn = (uint8_t)min_int;
    uint8_t mx = (uint8_t)max_int;
    float scale_constant = 255.0f / (mx - mn);

    // Kernels 2 and 3.
    sub_kernel<<<grid_size, block_size>>>(d_image, mn, n_pixels);
    CUDA_CHECK(cudaGetLastError());
    scale_kernel<<<grid_size, block_size>>>(d_image, scale_constant, n_pixels);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(e_kernels_end));
    // ---- end kernels-only interval ----

    // D2H: pull modified pixels back so the host can write the BMP.
    CUDA_CHECK(cudaMemcpy(out, d_image, img_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(e_total_end));
    // ---- end total interval ----

    // Wait for the very last event before reading elapsed times.
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
    // The four sample images supplied with the assignment.
    const char* samples[] = {
        "./samples/640x426.bmp",
        "./samples/1280x843.bmp",
        "./samples/1920x1280.bmp",
        "./samples/5184x3456.bmp",
    };
    const int n_samples = sizeof(samples) / sizeof(samples[0]);

    // Number of timed iterations to AVERAGE per image. Wall-clock noise on a
    // single run is large (Colab VMs are shared); averaging stabilizes things.
    // We also do ONE warm-up iteration that we throw away — the very first
    // CUDA call in a process pays JIT + context-init costs that aren't
    // representative of steady-state performance.
    const int n_warmup = 1;
    const int n_iters  = 5;

    // Two table rows per image: one using the atomic kernel (4a), one using
    // the shared-memory reduction kernel (4b). The subtract and scale kernels
    // are identical in both rows — the only thing that varies is KERNEL 1.
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

        uint8_t* buf_cpu = (uint8_t*)malloc(img_bytes);
        uint8_t* buf_gpu = (uint8_t*)malloc(img_bytes);

        // Run BOTH min/max variants. Each gets its own warmup + timed runs.
        const MinMaxKind variants[2]  = { MINMAX_ATOMIC, MINMAX_REDUCE };
        const char*      var_label[2] = { "atomic",      "reduce"      };

        // We average CPU time across all iterations of all variants — the CPU
        // pipeline doesn't depend on the GPU choice.
        float cpu_sum_all = 0.0f;
        int cpu_count_all = 0;

        // Per-variant accumulators.
        float gpu_total[2] = {0.0f, 0.0f};
        float gpu_kern[2]  = {0.0f, 0.0f};
        bool  match[2]     = {false, false};
        uint8_t mn_v[2] = {0,0}, mx_v[2] = {0,0};

        for (int v = 0; v < 2; v++) {
            // Warm-up.
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

            // Step 6 byte-identical correctness check: CPU result vs GPU result
            // for this variant. Both should be exactly the same bytes because
            // both pipelines use identical integer arithmetic and identical
            // float -> uint8_t truncation.
            match[v] = (memcmp(buf_cpu, buf_gpu, img_bytes) == 0);

            // Save outputs from the largest image for the report:
            //   out_img_cpu.bmp     - CPU baseline result
            //   out_img_gpu.bmp     - GPU result (using the proper reduction)
            // You can open them side by side, or run `cmp` on them to confirm
            // they are identical byte-for-byte.
            if (s == n_samples - 1 && variants[v] == MINMAX_REDUCE) {
                stbi_write_bmp("./out_img_gpu.bmp", width, height, 1, buf_gpu);
                stbi_write_bmp("./out_img_cpu.bmp", width, height, 1, buf_cpu);
            }
        }
        const float cpu_ms = cpu_sum_all / cpu_count_all;

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
        // Show the min/max values found (identical between CPU/GPU and between
        // variants — printing once per image is enough). Useful for the report.
        printf("%-22s | min=%u max=%u (used by all variants)\n",
               "", (unsigned)mn_v[1], (unsigned)mx_v[1]);

        free(buf_cpu);
        free(buf_gpu);
        stbi_image_free(original);
    }

    printf("\nWrote ./out_img_cpu.bmp and ./out_img_gpu.bmp for visual comparison.\n");
    printf("Verify byte-identical with:  cmp out_img_cpu.bmp out_img_gpu.bmp\n");
    return 0;
}
