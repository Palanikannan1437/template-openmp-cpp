#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1000

int main(int argc, char **argv) {
  int rank, size;
  int i, sum = 0;
  int array[ARRAY_SIZE];
  double start_time, end_time, time_diff;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    for (i = 0; i < ARRAY_SIZE; i++) {
      array[i] = i;
    }
  }

  start_time = MPI_Wtime();

  int local_array[ARRAY_SIZE / size];
  MPI_Scatter(array, ARRAY_SIZE / size, MPI_INT, local_array, ARRAY_SIZE / size,
              MPI_INT, 0, MPI_COMM_WORLD);

  for (i = 0; i < ARRAY_SIZE / size; i++) {
    sum += local_array[i];
  }

  int global_sum;
  MPI_Gather(&sum, 1, MPI_INT, &global_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);

  printf("My registration number is 20BCE2488 and my name is M.Palanikannan\n");
  if (rank == 0) {
    double average = (double)global_sum / ARRAY_SIZE;
    printf("Average: %f\n", average);
  }

  end_time = MPI_Wtime();
  time_diff = end_time - start_time;

  if (rank == 0) {
    printf("Time taken: %f seconds\n", time_diff);
  }

  MPI_Finalize();
  return 0;
}