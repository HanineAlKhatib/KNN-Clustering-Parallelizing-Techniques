#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define N 1000 // Number of data points
#define DIM 32 // Dimensionality of data points
#define K 3    // Number of nearest neighbors to consider

// Calculate the Euclidean distance between two points
double euclidean_distance(double a[DIM], double b[DIM])
{
    double dist = 0.0;
    for (int i = 0; i < DIM; i++)
    {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

// Find the k-nearest neighbors for each point in the dataset
void knn(double dataset[N][DIM], int neighbors[N][K], int num_threads)
{
    int i, j;

    omp_set_num_threads(num_threads);

#pragma omp parallel for private(j) schedule(dynamic)
    for (i = 0; i < N; i++)
    {
        double dists[N];
        int sorted_indices[N];

        // Calculate the distances and keep track of the sorted indices
        for (j = 0; j < N; j++)
        {
            dists[j] = euclidean_distance(dataset[i], dataset[j]);
            sorted_indices[j] = j;
        }

        // Sort the distances using Bubble Sort (use an efficient sorting algorithm for large datasets)
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

        // Store the indices of the k-nearest neighbors
        for (int k = 0; k < K; k++)
        {
            neighbors[i][k] = sorted_indices[k + 1]; // Skip the first index as it is the point itself
        }
    }
}

// Load dataset from the CSV file
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
    double dataset[N][DIM];
    int neighbors[N][K];
    int num_threads;

    load_dataset("KNNAlgorithmDataset.csv", dataset);

    printf("Enter the number of threads: ");
    scanf("%d", &num_threads);
    getchar(); // Capture the newline character from the input buffer

    clock_t start = clock();
    knn(dataset, neighbors, num_threads);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    // Print the k-nearest neighbors for each point
    for (int i = 0; i < N; i++)
    {
        printf("Neighbors of point %d: ", i);
        for (int j = 0; j < K; j++)
        {
            printf("%d ", neighbors[i][j]);
        }
        printf("\n");
    }

    printf("Execution time: %.5f seconds\n", time_taken);

    return 0;
}
