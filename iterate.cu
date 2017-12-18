__global__
void iterateKernel(int maxIterations, double xOrigin, double yOrigin, double zoomFactor, int* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int p = index; p < 360000; p += stride) {
        // deliniarize
        int i = p / 600;
        int j = p % 600;

        // convert to complex number
        double cx = xOrigin - (2 / zoomFactor) * (1 - 2 * ((double) i / 600));
        double cy = yOrigin - (2 / zoomFactor) * (1 - 2 * ((double) j / 600));

        // do the iterations
        double zx = cx;
        double zy = cy;
        double tx;
        double ty;
        bool inMandelbrot = true;
        for(int k = 0; k < maxIterations; ++ k)
        {
            if(zx * zx + zy * zy > 4) {
                result[i*600+j] = 255 * (1 - (double) k / maxIterations);
                inMandelbrot = false;
                break;
            }
            tx = zx * zx - zy * zy + cx;
            ty = 2 * zx * zy + cy;
            zx = tx;
            zy = ty;
        }
        if(inMandelbrot)
            result[i*600+j] = 0;
    }
}

extern "C"
int* iterateGPU(int maxIterations, double xOrigin, double yOrigin, double zoomFactor) {
    int* resultOnGPU;
    cudaMalloc(&resultOnGPU, 600 * 600 * sizeof(int));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = deviceProp.maxThreadsPerBlock;
    int numBlocks = (600 * 600 - 1) / blockSize + 1;

    iterateKernel<<<numBlocks, blockSize>>>(maxIterations, xOrigin, yOrigin, zoomFactor, resultOnGPU);
    cudaDeviceSynchronize();

    auto result = (int*) malloc(600 * 600 * sizeof(int));
    cudaMemcpy(result, resultOnGPU, 600 * 600 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(resultOnGPU);

    return result;
}
