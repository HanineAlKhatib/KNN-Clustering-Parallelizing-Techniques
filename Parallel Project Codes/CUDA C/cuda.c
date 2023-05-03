#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000 // Number of data points
#define DIM 32 // Dimensionality of data points
#define K 3    // Number of nearest neighbors to consider

__global__ void knn_kernel(double *dataset, int *neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        double dists[N];
        int sorted_indices[N];

        for (int j = 0; j < N; j++)
        {
            double dist = 0.0;
            for (int d = 0; d < DIM; d++)
            {
                double diff = dataset[i * DIM + d] - dataset[j * DIM + d];
                dist += diff * diff;
            }
            dists[j] = sqrt(dist);
            sorted_indices[j] = j;
        }

        for (int m = 0; m < K; m++)
        {
            for (int n = 0; n < N - m - 1; n++)
            {
                if (dists[sorted_indices[n]] > dists[sorted_indices[n + 1]])
                {
                    int temp = sorted_indices[n];
                    sorted_indices[n] = sorted_indices[n + 1];
                    sorted_indices[n + 1] = temp;
                }
            }
        }

        for (int k = 0; k < K; k++)
        {
            neighbors[i * K + k] = sorted_indices[k + 1];
        }
    }
}

void load_dataset(const char *filename, double dataset[N][DIM])
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        exit(1);
    }

    char line[1024];
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < N)
    {
        char *token = strtok(line, ",");
        for (int j = 0; j < DIM; j++)
        {
            if (token != NULL)
            {
                dataset[i][j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
        i++;
    }

    fclose(file);
}

int main()
{
    double h_dataset[N][DIM];
    int h_neighbors[N][K];
    double *d_dataset;
    int *d_neighbors;

    size_t dataset_size = N * DIM * sizeof(double);
    size_t neighbors_size = N * K * sizeof(int);

    load_dataset("KNNAlgorithmDataset.csv", h_dataset);

    cudaMalloc((void **)&d_dataset, dataset_size);
    cudaMalloc((void **)&d_neighbors, neighbors_size);

    cudaMemcpy(d_dataset, h_dataset, dataset_size, cudaMemcpyHostToDevice);

    int blockSize = 256; // Fixed block size
    int numThreads, gridSize;
    printf("Enter the number of threads: ");
    scanf("%d", &numThreads);
    gridSize = (numThreads + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    knn_kernel<<<gridSize, blockSize>>>(d_dataset, d_neighbors);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_neighbors, d_neighbors, neighbors_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("Neighbors of point %d: ", i);
        for (int j = 0; j < K; j++)
        {
            printf("%d ", h_neighbors[i][j]);
        }
        printf("\n");
    }

    printf("Execution time: %.5f milliseconds\n", elapsedTime);

    cudaFree(d_dataset);
    cudaFree(d_neighbors);

    return 0;
}
