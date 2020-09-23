#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

int recursion_add(int *a, int len, int *s)
{
    if (len == 1)
    {
        return 1;
    }
    memset(s, 0, sizeof(int) * len);
    if (len % 2)
    {
#pragma omp parallel for
        for (int i = 0; i < len / 2; i++)
        {
            s[i] = a[i] + a[i + len / 2];
        }
        s[len / 2] = a[len - 1];
        recursion_add(s, len / 2 + 1, a);
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < len / 2; i++)
        {
            s[i] = a[i] + a[i + len / 2];
        }
        recursion_add(s, len / 2, a);
    }
    return 0;
}

int main(int argc, char **argv)
{
    int processor_num = omp_get_num_procs();
    printf("The CPU has %d threads\n", processor_num);
    if (argc<2)
    {
        printf("Please input the size of the data and the thread set for OpenMp\n");
        return -1;
    }
    

    struct timeval tv1, tv2;
    int n = atoi(argv[1]);
    int thread_num = atoi(argv[2]);
    printf("We set %d threads for OpenMP\n", thread_num);

    int *a = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++)
    {
        a[i] = i + 1;
    }
    gettimeofday(&tv1, NULL);

    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    gettimeofday(&tv2, NULL);
    double serial_time = (tv2.tv_sec - tv1.tv_sec) * 1000.0 + (tv2.tv_usec - tv1.tv_usec) / 1000.0;

    sum = 0;
#pragma omp parallel for shared(sum)
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    gettimeofday(&tv2, NULL);
    double omp_serial_time = (tv2.tv_sec - tv1.tv_sec) * 1000.0 + (tv2.tv_usec - tv1.tv_usec) / 1000.0;

    sum = 0;
#pragma omp parallel for reduction(+ \
                                   : sum)
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    gettimeofday(&tv2, NULL);
    double omp_reduction_time = (tv2.tv_sec - tv1.tv_sec) * 1000.0 + (tv2.tv_usec - tv1.tv_usec) / 1000.0;

    int sum_omp[thread_num];
    memset(sum_omp, 0, sizeof(int) * thread_num);
    gettimeofday(&tv1, NULL);
#pragma omp parallel for
    for (int i = 0; i < thread_num; i++)
    {
        int block_size = n / thread_num;
        int start = i * block_size;
        for (int j = 0; j < block_size; j++)
        {
            sum_omp[i] += a[start + j];
        }
    }
    for (int i = 1; i < thread_num; i++)
    {
        sum_omp[0] += sum_omp[i];
        if (i < n % thread_num)
        {
            sum_omp[0] += a[n - 1 - i];
        }
    }

    gettimeofday(&tv2, NULL);
    double omp_time = (tv2.tv_sec - tv1.tv_sec) * 1000.0 + (tv2.tv_usec - tv1.tv_usec) / 1000.0;

    int *s = (int *)malloc(sizeof(int) * n);
    memset(s, 0, sizeof(int) * n);
    gettimeofday(&tv1, NULL);
    int len = recursion_add(a, n, s);
    gettimeofday(&tv2, NULL);
    double recursion_time = (tv2.tv_sec - tv1.tv_sec) * 1000.0 + (tv2.tv_usec - tv1.tv_usec) / 1000.0;

    printf("the result is             : %d\n", a[0] > s[0] ? a[0] : s[0]);
    printf("the Serial cost           : %.2lfms\n", serial_time);
    printf("the OpenMp Serial cost    : %.2lfms\n", omp_serial_time);
    printf("the OpenMp Reduction cost : %.2lfms\n", omp_reduction_time);
    printf("the OpenMp cost           : %.2lfms\n", omp_time);
    printf("the Recursion cost        : %.2lfms\n", recursion_time);

    free(a);
    return 0;
}