#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

#define  N   (64)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;

double eps;
double A [N][N][N];

void relax();
void init();
void verify(); 

int main(int an, char **as)
{
    if (an < 2) {
		printf("Bad arguments!\n");
		return -1;
	}
	int it;
    int num_thr = atoi(as[1]);

	init();
    double time1 = omp_get_wtime();

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax(num_thr);
		if (eps < maxeps) break;
	}

    double time2 = omp_get_wtime();

	verify();

	printf("time = %f\n", time2 - time1);

	return 0;
}


void init()
{ 
	for(i=0; i<=N-1; i++) {
	    for(j=0; j<=N-1; j++) {
	        for(k=0; k<=N-1; k++)
            {
                if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
                A[i][j][k]= 0.;
                else A[i][j][k]= ( 4. + i + j + k) ;
            }
        }
    }
} 

void relax(int num_thr){
#pragma omp parallel shared(A, eps) num_threads(num_thr)
{
    double local_eps = eps;
	#pragma omp for 

    for(i=1; i<=N-2; i++)
    for(j=1; j<=N-2; j++)
    for(k=1; k<=N-2; k++)
    {
        A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k])/2.;
        local_eps = Max(fabs(A[i][j][k]),local_eps);
    }
    #pragma omp for

    for(i=1; i<=N-2; i++)
    for(j=1; j<=N-2; j++)
    for(k=1; k<=N-2; k++)
    {
        A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k])/2.;
    }
    #pragma omp for

    for(i=1; i<=N-2; i++)
    for(j=1; j<=N-2; j++)
    for(k=1; k<=N-2; k++)
    {
        A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1])/2.;
    }
    #pragma omp critical
	{
		eps = Max(eps, local_eps);
	}

    }


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