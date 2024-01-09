#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "mpi-ext.h"

#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (100)

double   maxeps = 0.1e-7;
int itmax = 7;
int i, j, k;

int ranksize, myrank;
int min_str = 0, max_str = N - 1;

double eps;

double A[N][N][N];

void relax();
void init();
void verify();

int get_minstr(int, int);
int get_maxstr(int, int);

void write_checkpoint(int it);
void read_checkpoint(int *it);

int malloc3d(double ****array, int dim1, int dim2, int dim3) {
    double *p = (double *)malloc(dim1 * dim2 * dim3 * sizeof(double));
    if (!p) return -1;

    (*array) = (double ***)malloc(dim1 * sizeof(double*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    for (int i = 0; i < dim1; i++) {
        (*array)[i] = (double **)malloc(dim2 * sizeof(double*));
        if (!(*array)[i]) {
            for (int j = 0; j < i; j++) {
                free((*array)[j]);
            }
            free(*array);
            free(p);
            return -1;
        }
        for (int j = 0; j < dim2; j++) {
            (*array)[i][j] = &(p[(i * dim2 + j) * dim3]);
        }
    }

    return 0;
}

int main(int an, char **as)
{
    double t1, t2;
    MPI_Init(&an, &as);

    int it;
    int failed_size;

    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0)
    {
        t1 = MPI_Wtime();
    }

    min_str = get_minstr(ranksize, myrank);
    max_str = get_maxstr(ranksize, myrank);
    double ***A;
    malloc3d(&A, max_str - min_str, N, N);
    init();

    for (it = 1; it <= itmax; it++)
    {
        eps = 0.;
        relax();
        MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (eps < maxeps)
            break;
        write_checkpoint(it);
        
        if (myrank == 0)
            {
                int failed_rank = rand() % ranksize;
                if (failed_rank != 0)
                {
                    printf("Process %d fails at iteration %d\n", failed_rank, it);
                    failed_size++;

                }
            }  

        read_checkpoint(&it);
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0)
    {
        t2 = MPI_Wtime();
        printf("time  = %f\n", t2 - t1);
        // verify();
    }


    // Выявление сбоев
    MPIX_Comm_failure_ack(MPI_COMM_WORLD);
    MPI_Group group_world, group_failed;
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    MPIX_Comm_failure_get_acked(MPI_COMM_WORLD, &group_failed);

    MPI_Group_size(group_failed, &failed_size);

    // Исключение процессов, в которых произошел сбой
    MPI_Comm new_comm;
    MPIX_Comm_shrink(MPI_COMM_WORLD, &new_comm);

    MPI_Comm_rank(new_comm, &myrank);
    MPI_Comm_size(new_comm, &ranksize);


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) 
        {
            free(A[i][j]);
        }
        free(A[i]);
    }
    free(A);

    MPI_Finalize();

    return 0;
}

void init()
{ 
	for(i=0; i<=max_str - 1 - min_str; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                if(i + min_str == 0 || i + min_str == N-1 || j==0 || j==N-1 || k==0 || k==N-1)
                    A[i][j][k]= 0.;
                else A[i][j][k]= ( 4. + i + j + k + min_str) ;
            }
} 

void relax()
{
    MPI_Status status;
	MPI_Request request;
	int nextrank = myrank == ranksize - 1 ? MPI_PROC_NULL : myrank + 1;
	int prevrank = myrank == 0 ? MPI_PROC_NULL : myrank - 1;

	for(i=1; i<=max_str - 1 - min_str; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k])/2.;
                eps = Max(fabs(A[i][j][k]),eps);
            }

    MPI_Isend(A[max_str - 1 - min_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[max_str - min_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &status);
	
	MPI_Isend(A[1], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[0], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &status);

	for(i=1; i<=max_str - 1 - min_str; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k])/2.;
            }

    MPI_Isend(A[max_str - 1 - min_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[max_str - min_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &status);
	
	MPI_Isend(A[1], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[0], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &status);

	for(i=1; i<=max_str - 1 - min_str; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1])/2.;
            }

	MPI_Isend(A[max_str - 1 - min_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[max_str - min_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &status);

	MPI_Isend(A[1], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[0], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &status);

}

/*
void verify()
{
	double s;

	s=0.;
	for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
            for(k=0; k<=N-1; k++)
            {
                s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
            }
	printf("  S = %f\n",s);

}
*/

int get_maxstr(int num, int rank){
	if (rank == num - 1){
		return N-1;
	} else {
		return (N / num)*(rank + 1);
	}
}

int get_minstr(int num, int rank){
	if (rank == 0){
		return 0;
	} else {
		return N / num * rank - 1;
	}
}

void write_checkpoint(int it)
{
    char filename[20];
    sprintf(filename, "checkpoint_%d.txt", it);
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_File_write_at(fh, min_str * N * N * sizeof(A[0][0][0]), &A[0][0][0], (max_str - min_str) * N * N, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
}

void read_checkpoint(int *it)
{
    char filename[20];
    sprintf(filename, "checkpoint_%d.txt", *it);
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_File_read_at(fh, min_str * N * N * sizeof(A[0][0][0]), &A[0][0][0], (max_str - min_str) * N * N, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
}