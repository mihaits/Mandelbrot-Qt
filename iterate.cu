__global__
void iterateKernel(int w, int h, int maxIterations, double xOrigin, double yOrigin, double zoomFactor, int* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int p = index; p < w * h; p += stride) {
        // deliniarize
        int i = p / w;
        int j = p % w;

        // convert to complex number
        double cx = xOrigin - (2 / zoomFactor) * (1 - 2 * ((double) j / w));
        double cy = yOrigin - (2 / zoomFactor) * (1 - 2 * ((double) (i+(w-h)/2) / w));

        // do the iterations
        double zx = cx;
        double zy = cy;
        double tx;
        double ty;
        bool inMandelbrot = true;
        for(int k = 0; k < maxIterations; ++ k)
        {
            if(zx * zx + zy * zy > 4) {
                result[i*w+j] = 255 * (1 - (double) k / maxIterations);
                inMandelbrot = false;
                break;
            }
            tx = zx * zx - zy * zy + cx;
            ty = 2 * zx * zy + cy;
            zx = tx;
            zy = ty;
        }
        if(inMandelbrot)
            result[i*w+j] = 0;
    }
}

extern "C"
int* iterateGPU(int w, int h, int maxIterations, double xOrigin, double yOrigin, double zoomFactor) {
    int* resultOnGPU;
    cudaMalloc(&resultOnGPU, w * h * sizeof(int));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = deviceProp.maxThreadsPerBlock;
    int numBlocks = (w * h - 1) / blockSize + 1;

    iterateKernel<<<numBlocks, blockSize>>>(w, h, maxIterations, xOrigin, yOrigin, zoomFactor, resultOnGPU);
    cudaDeviceSynchronize();

    auto result = (int*) malloc(w * h * sizeof(int));
    cudaMemcpy(result, resultOnGPU, w * h * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(resultOnGPU);

    return result;
}
