#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
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
void knn(double dataset[N][DIM], int neighbors[N][K], int start_idx, int end_idx)
{
    int i, j;

    for (i = start_idx; i < end_idx; i++)
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

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double dataset[N][DIM];
    int neighbors[N][K];

    if (rank == 0)
    {
        load_dataset("KNNAlgorithmDataset.csv", dataset);
    }

    MPI_Bcast(dataset, N * DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int start_idx = rank * N / size;
    int end_idx = (rank + 1) * N / size;

    double start = MPI_Wtime();
    knn(dataset, neighbors, start_idx, end_idx);
    double end = MPI_Wtime();
    double time_taken = end - start;

    // Gather the nearest neighbors from all processes
    int local_neighbors_count = (end_idx - start_idx) * K;
    int recvcounts[size];
    int displs[size];
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        recvcounts[i] = (N / size) * K;
        displs[i] = sum;
        sum += recvcounts[i];
    }

    MPI_Gatherv(&neighbors[start_idx], local_neighbors_count, MPI_INT,
     neighbors, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the k-nearest neighbors for each point and execution time
    if (rank == 0)
    {
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
    }

    MPI_Finalize();

    return 0;
}
