#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (200)
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
	double time_spent = 0.0; // Для хранения времени выполнения
	clock_t begin = clock();

	int it;
	init();

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		// printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	verify();

	clock_t end = clock();

	// рассчитать прошедшее время, найдя разницу (end - begin) и
    // деление разницы на CLOCKS_PER_SEC для перевода в секунды

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("The elapsed time is %f seconds", time_spent);

	return 0;
}


void init()
{ 
	for(k=0; k<=N-1; k++)
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
		A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
} 

void relax()
{
	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k])/2.;
	}

	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k])/2.;
	}

	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		double e;
		e=A[i][j][k];
		A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1])/2.;
		eps=Max(eps,fabs(e-A[i][j][k]));
	}

}

void verify()
{
	double s;

	s=0.;
	for(k=0; k<=N-1; k++)
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	printf("  S = %f\n",s);

}
