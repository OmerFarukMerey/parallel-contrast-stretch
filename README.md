# parallel-contrast-stretch

CUDA-accelerated automatic contrast stretching for grayscale BMP images, with a CPU baseline for comparison.

The pipeline rescales pixel values so the darkest pixel becomes 0 and the brightest becomes 255:

```
output = (input - min) * 255 / (max - min)
```

Three GPU kernels do the work:

1. **min/max reduction** — a tree reduction in shared memory finds the minimum and maximum pixel values. One atomic update per block instead of one per pixel.
2. **subtract** — one thread per pixel subtracts `min`.
3. **scale** — one thread per pixel multiplies by `255 / (max - min)`.

A naive global-atomic version of the min/max kernel is also included so you can see how much shared-memory reduction buys you.

## Layout

```
main.cpp     CPU-only reference implementation
main.cu     GPU implementation + benchmark harness
samples/    Test images at four resolutions (640x426 -> 5184x3456)
stb_image*  Single-header image I/O (public domain)
```

## Build

```sh
nvcc -O2 main.cu -o contrast_stretch
```

CPU-only baseline:

```sh
g++ -O2 main.cpp -o contrast_stretch_cpu
```

## Run

```sh
./contrast_stretch
```

The benchmark loads every image in `samples/`, runs both the CPU and GPU pipelines (with one warm-up and five timed iterations each), and prints a table comparing wall-clock time, kernel-only time, and end-to-end speedup. It also writes `out_img_cpu.bmp` and `out_img_gpu.bmp` from the largest sample so you can visually confirm the results match.

To verify the outputs are byte-identical:

```sh
cmp out_img_cpu.bmp out_img_gpu.bmp
```

## Notes

- The GPU pipeline uses `cudaEvent`s to time kernels separately from H2D/D2H transfers. On small images PCIe transfers dominate; on large images the kernels dominate and the speedup grows.
- All arithmetic uses the same integer math and `float -> uint8_t` truncation as the CPU baseline so outputs match exactly.
- Tested on Google Colab T4 / L4 GPUs.
