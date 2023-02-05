#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (100)

double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;

int ranksize, myrank;
int min_str = 0, max_str = N-1;

double eps;

double A [N][N][N];

void relax();
void init();
void verify(); 

int get_minstr(int, int);
int get_maxstr(int, int);


int main(int an, char **as)
{
    	double t1, t2;
	MPI_Init(&an, &as);

	int it;
	init();

    	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Barrier(MPI_COMM_WORLD);
	if (myrank == 0){
		t1 = MPI_Wtime();
	}

	min_str = get_minstr(ranksize, myrank);
	max_str = get_maxstr(ranksize, myrank);

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if (eps < maxeps) 
            break;
	}

    MPI_Barrier(MPI_COMM_WORLD);
	if (myrank == 0){
		t2 = MPI_Wtime();
		printf("time  = %f\n", t2 - t1);
		verify();
    }

    MPI_Finalize();

	return 0;
}


void init()
{ 
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
		A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
} 

void relax()
{
    	MPI_Status status;
	MPI_Request request;
	int nextrank = myrank == ranksize - 1 ? MPI_PROC_NULL : myrank + 1;
	int prevrank = myrank == 0 ? MPI_PROC_NULL : myrank - 1;

	for(i=min_str + 1; i<=max_str - 1; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k])/2.;
        eps = Max(fabs(A[i][j][k]),eps);
	}

    	MPI_Isend(A[max_str-1], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[max_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &status);
	
	MPI_Isend(A[min_str + 1], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[min_str], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &status);

	for(i=min_str + 1; i<=max_str - 1; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k])/2.;
	}

    	MPI_Isend(A[max_str-1], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[max_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &status);
	
	MPI_Isend(A[min_str + 1], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[min_str], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &status);

	for(i=min_str + 1; i<=max_str - 1; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1])/2.;
	}

	MPI_Isend(A[max_str - 1], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[max_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &status);

	MPI_Isend(A[min_str + 1], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[min_str], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &status);

}

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
