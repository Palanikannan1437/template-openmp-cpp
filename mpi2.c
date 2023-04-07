#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("My registration number is 20BCE2488 and my name is M.Palanikannan\n");

  int n = 1000000;
  int *data = NULL;
  double t1, t2;

  if (rank == 0) {
    data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
      data[i] = rand() % 100;
    }
  }

  int local_n = n / size;
  int *local_data = (int*)malloc(local_n * sizeof(int));

  t1 = MPI_Wtime();
  MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

  int local_sum = 0;
  for (int i = 0; i < local_n; i++) {
    local_sum += local_data[i];
  }

  int global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Average = %.2f\n", (double)global_sum / n);
    t2 = MPI_Wtime();
    printf("Time = %.6f\n", t2 - t1);
    free(data);
  }

  free(local_data);
  MPI_Finalize();
  return 0;
}
