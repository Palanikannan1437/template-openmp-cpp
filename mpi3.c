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
  int *data = (int*)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    data[i] = rand() % 100;
  }

  int *local_data = (int*)malloc(n / size * sizeof(int));
  MPI_Scatter(data, n / size, MPI_INT, local_data, n / size, MPI_INT, 0, MPI_COMM_WORLD);

  double t1 = MPI_Wtime();

  // Approach 1: MPI_Bcast
  MPI_Bcast(data, n, MPI_INT, 0, MPI_COMM_WORLD);

  // Perform collective communication operation (MPI_Allreduce in this case)
  int local_sum = 0;
  for (int i = 0; i < n / size; i++) {
    local_sum += local_data[i];
  }
  int global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  double t2 = MPI_Wtime();

  if (rank == 0) {
    printf("Global sum = %d\n", global_sum);
    printf("Time with MPI_Bcast = %.6f\n", t2 - t1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  t1 = MPI_Wtime();

  // Approach 2: MPI_Send and MPI_Recv
  if (rank == 0) {
    for (int i = 1; i < size; i++) {
      MPI_Send(data, n, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_data, n / size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Perform collective communication operation (MPI_Allreduce in this case)
  local_sum = 0;
  for (int i = 0; i < n / size; i++) {
    local_sum += local_data[i];
  }
  global_sum = 0;
  for (int i = 0; i < size; i++) {
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, i, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  t2 = MPI_Wtime();

  if (rank == 0) {
    printf("Global sum = %d\n", global_sum);
    printf("Time with MPI_Send and MPI_Recv = %.6f\n", t2 - t1);
  }

  free(data);
  free(local_data);
  MPI_Finalize();
  return 0;
}
