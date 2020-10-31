#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

//Output the matrix A
void Print(int n, float *A)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%2.1f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("**************************************************\n");
}

//Check the result
void Check(int n, float *A)
{
    for (int i = 0; i < n; i++)
    {
        int j = 0;
        for (; j <= i; j++)
        {
            if (A[i * n + j] != 1 && A[i * n + j] != 0)
            {
                printf("Matrix factorization failed.\n");
                return;
            }
        }
    }
    printf("Matrix factorization succeeded.\n");
    return;
}

float InvSqrt(float x)
{
    float xhalf = 0.5f * x;
    int i = *(int *)&x;
    i = 0x5f3759df - (i >> 1); // 计算第一个近似根
    x = *(float *)&i;
    x = x * (1.5f - xhalf * x * x); // 牛顿迭代法
    return x;
}

//Cholesky factorization
void Cholesky(int n, float *A)
{
    for (int a = 0; a < n; a++)
    {
        float sum = 0;
        for (int i = 0; i < a; i++)
            sum += A[a * n + i] * A[a * n + i];
        sum = A[a * n + a] - sum;
        // A[a * n + a] = InvSqrt(sum > 0 ? sum : 0);
        A[a * n + a] = (sum > 0 ? sqrt(sum) : 0);
#pragma omp parallel for
        for (int i = a + 1; i < n; i++)
        {
            sum = 0;
            for (int j = 0; j < a; j++)
                sum += A[i * n + j] * A[a * n + j];
            A[i * n + a] = (A[i * n + a] - sum) / A[a * n + a];
        }
#pragma omp parallel for
        for (int j = 0; j < a; j++)
            A[j * n + a] = 0;
    }
}

int main(int argc, const char *argv[])
{
    struct timeval start, end;
    int n = 8000;
    printf("******************** %d*%d Matrix *****************\n", n, n);

    gettimeofday(&start, NULL);
    float *A = (float *)malloc(sizeof(float) * n * n);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        int j = 0;
        for (; j <= i; j++)
            A[i * n + j] = j + 1;
        for (; j < n; j++)
            A[i * n + j] = i + 1;
    }
    gettimeofday(&end, NULL);
    double time_generate = 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000;
    printf("the time of Cholesky generation is %.2lf ms\n", time_generate);

    //Do cholesky factorization
    gettimeofday(&start, NULL);
    Cholesky(n, A);
    gettimeofday(&end, NULL);

    double all_time = 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000;
    printf("the time of Cholesky factorization is %.2lf ms\n", all_time);

    //check the result
    Check(n, A);

    //You can use the statement below to output the decomposed matrix
    //Print(n, A);
    free(A);

    return 0;
}
