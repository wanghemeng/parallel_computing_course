#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include <math.h>
#include "mmio.h"
#include "mmiohighlevel.h"
#include "omp.h"

typedef struct
{
    VALUE_TYPE *value;
    int *columnindex;
    int *rowpointer;

} SMatrix;

SMatrix Spgmm_to_complexCSR(VALUE_TYPE *A, int n, int m);
SMatrix Spgmm_to_complexCSC(VALUE_TYPE *A, int n, int m);
void CSR_to_CSC(SMatrix A, SMatrix *B, int n, int m);
void display_SMatrix(SMatrix A, int n);
void display_special_SMtrix(SMatrix A, int n, int m);
void multiply_SpecialSMatrixCSR_with_SMatrixCSC_get_SpecialSMatrixCSR(SMatrix A, SMatrix B, SMatrix C, int n, int m, int k, double bias);
int init_SpecialSMatrix(SMatrix *A, int n, int m);
void check(FILE *fs, SMatrix A, int n, int m);
int NormalSMatrix_to_SpecialMatrix(SMatrix A, SMatrix *B, int n, int m);

SMatrix Spgmm_to_complexCSR(VALUE_TYPE *A, int n, int m)
{
    int zeronum = 0;
    SMatrix B;
    B.value = malloc(sizeof(VALUE_TYPE) * n * m);
    B.columnindex = malloc(sizeof(int) * (n + 1));
    B.rowpointer = malloc(sizeof(int) * n * m);
    int *memory = malloc(sizeof(int) * (n + 1));
    memset(B.rowpointer, 0, sizeof(int) * (n + 1));
    memset(memory, 0, sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (fabs(A[i * m + j]) > 0.01)
            {
                B.value[memory[i]] = A[i * m + j];
                B.columnindex[memory[i]] = j;
                memory[i]++;
            }
        }
        memory[i + 1] = memory[i];
    }
    for (int i = 0; i < n; i++)
    {
        B.rowpointer[i + 1] = memory[i];
    }
    return B;
}

SMatrix Spgmm_to_complexCSC(VALUE_TYPE *A, int n, int m)
{
    int zeronum = 0;
    SMatrix B;
    B.value = malloc(sizeof(VALUE_TYPE) * n * m);
    B.columnindex = malloc(sizeof(int) * (n + 1));
    B.rowpointer = malloc(sizeof(int) * n * m);
    memset(B.rowpointer, 0, sizeof(int) * (n + 1));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (fabs(A[i * n + j]) > 0.01)
            {
                B.value[B.rowpointer[i]] = A[i * n + j];
                B.columnindex[B.rowpointer[i]] = j;
                B.rowpointer[i]++;
            }
        }
        B.rowpointer[i + 1] = B.rowpointer[i];
    }
    return B;
}
void CSR_to_CSC(SMatrix A, SMatrix *B, int n, int m)
{
    // B->rowpointer = malloc(sizeof(int) * (m + 1));
    // B->columnindex = malloc(sizeof(int) * A.rowpointer[n + 1]);
    // B->value = malloc(sizeof(VALUE_TYPE) * (A.rowpointer[n + 1]));
    int *memory = (int *)malloc(sizeof(int) * (m + 1));
    memset(B->rowpointer, 0, sizeof(int) * (m + 1));
    memset(memory, 0, sizeof(int) * (m + 1));
    for (int i = 0; i < A.rowpointer[n]; i++)
    {
        B->rowpointer[A.columnindex[i] + 1]++;
    }
    for (int i = 0; i < m; i++)
    {
        B->rowpointer[i + 1] += B->rowpointer[i];
        memory[i + 1] = B->rowpointer[i + 1];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = A.rowpointer[i]; j < A.rowpointer[i + 1]; j++)
        {
            B->columnindex[memory[A.columnindex[j]]] = i;
            B->value[memory[A.columnindex[j]]] = A.value[j];
            memory[A.columnindex[j]]++;
        }
    }
    free(memory);
}
void display_SMatrix(SMatrix A, int n)
{
    printf("this matrix is a normal SMatrix \n");
    printf("rowpointer: ");
    for (int i = 0; i < n + 1; i++)
    {
        printf("%d ", A.rowpointer[i]);
    }
    printf("\ncolumnindex: ");
    for (int i = 0; i < A.rowpointer[n]; i++)
    {
        printf("%d ", A.columnindex[i]);
    }
    printf("\nvalue: ");
    for (int i = 0; i < A.rowpointer[n]; i++)
    {
        printf("%f ", A.value[i]);
    }
    printf("\n");
}
void display_special_SMtrix(SMatrix A, int n, int m)
{
    printf("this matrix is a special SMatrix %d * %d\n", n, m);
    printf("rowpointer: ");
    for (int i = 0; i < n + 1; i++)
    {
        printf("%d ", A.rowpointer[i]);
    }
    printf("\ncolumnindex: ");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (A.rowpointer[i + 1] - A.rowpointer[i]); j++)
        {
            printf("%d ", A.columnindex[i * m + j]);
        }
    }
    printf("\nvalue: ");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (A.rowpointer[i + 1] - A.rowpointer[i]); j++)
        {
            printf("%f ", A.value[i * m + j]);
        }
    }
    printf("\n");
}
int NormalSMatrix_to_SpecialMatrix(SMatrix A, SMatrix *B, int n, int m)
{
    B->rowpointer = malloc(sizeof(int) * (n + 1));
    B->columnindex = malloc(sizeof(int) * n * m);
    B->value = malloc(sizeof(VALUE_TYPE) * n * m);
    for (int i = 0; i < n + 1; i++)
    {
        B->rowpointer[i] = A.rowpointer[i];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (A.rowpointer[i + 1] - A.rowpointer[i]); j++)
        {
            B->columnindex[i * m + j] = A.columnindex[A.rowpointer[i] + j];
            B->value[i * m + j] = A.value[A.rowpointer[i] + j];
        }
    }
    return 1;
}

int init_SpecialSMatrix(SMatrix *A, int n, int m)
{
    // SMatrix A;
    A->rowpointer = malloc(sizeof(int) * (n + 1));
    A->columnindex = malloc(sizeof(int) * n * m);
    A->value = malloc(sizeof(VALUE_TYPE) * n * m);
    memset(A->rowpointer, 0, n + 1);
    return 1;
}

void multiply_SpecialSMatrixCSR_with_SMatrixCSC_get_SpecialSMatrixCSR(SMatrix A, SMatrix B, SMatrix C, int n, int m, int k, double bias)
{
    // printf("n:%d m:%d k:%d\n", n, m, k);
    memset(C.rowpointer, 0, sizeof(int) * (n + 1));
    // printf("n:%d m:%d k:%d\n", n, m, k);

// #pragma omp parallel for schedule(dynamic)
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            // printf("i:%d j:%d \n", i, j);
            VALUE_TYPE sum = 0;
            for (int pointA = A.rowpointer[i], pointB = B.rowpointer[j]; (pointA < A.rowpointer[i + 1]) && (pointB < B.rowpointer[j + 1]);)
            {
                int colA = A.columnindex[pointA - A.rowpointer[i] + i * m];
                int rowB = B.columnindex[pointB];
                if (colA > rowB)
                {
                    pointB++;
                }
                else if (colA < rowB)
                {
                    pointA++;
                }
                else
                {
                    sum += (A.value[pointA - A.rowpointer[i] + i * m] * B.value[pointB]);
                    pointA++, pointB++;
                }
            }

            sum += bias;
            if (sum >= 0.0)
            {
                int index = C.rowpointer[i + 1] + i * k;
                C.columnindex[index] = j;
                if (sum >= 32.0)
                {
                    C.value[index] = 32.0;
                }
                else
                {
                    C.value[index] = sum;
                }
                C.rowpointer[i + 1]++;
            }
        }
    }
    // printf("pre\n");

    for (int i = 0; i < n; i++)
    {
        C.rowpointer[i + 1] += C.rowpointer[i];
    }
    // printf("num=%d\n", C.rowpointer[n]);
}

void copy_SpecialSMtrix(SMatrix *A, SMatrix *B, int n, int m)
{
    SMatrix C;
    C.columnindex = A->columnindex;
    C.rowpointer = A->rowpointer;
    C.value = A->value;
    A->columnindex = B->columnindex;
    A->rowpointer = B->rowpointer;
    A->value = B->value;
    B->columnindex = C.columnindex;
    B->rowpointer = C.rowpointer;
    B->value = C.value;
}

void check(FILE *fs, SMatrix A, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        // printf("%d %d %d\n", i, A.rowpointer[i + 1], A.rowpointer[i]);
        if (A.rowpointer[i + 1] != A.rowpointer[i])
        {
            fprintf(fs, "%d\n", i + 1);
        }
    }
}

int main(int argc, char **argv)
{
    // omp_set_num_threads(16);
    struct timeval t1, t2, t3, t4;
    double time_read_sum = 0, time_trans_sum = 0, time_compute_sum = 0, time_total = 0;
    int size1 = 0;
    int size2 = 0;
    int *tc1;
    int *tc2;
    double bias = -0.3000;

    int mA;
    int nA;
    int nnzA;
    int isSymmetricA;
    SMatrix A;
    SMatrix A_csr;

    int mB;
    int nB;
    int nnzB;
    int isSymmetricB;
    SMatrix B[120];

    int mC, nC;
    int nnzC_golden = 0;

    gettimeofday(&t1, NULL);

    // load matrix data from file
    char filename1[] = "sparse-images-1024.tsv";
    mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filename1);
    A_csr.value = (VALUE_TYPE *)malloc((nnzA) * sizeof(VALUE_TYPE));
    A_csr.columnindex = (int *)malloc((nnzA) * sizeof(int));
    A_csr.rowpointer = (int *)malloc((mA + 1) * sizeof(int));
    mmio_data(A_csr.rowpointer, A_csr.columnindex, A_csr.value, filename1);
    printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);
    NormalSMatrix_to_SpecialMatrix(A_csr, &A, mA, nA);

    free(A_csr.value);
    free(A_csr.columnindex);
    free(A_csr.rowpointer);

    char neuronfile1[] = "neuron1024/n1024-l";
    char neuronfile2[] = ".tsv";
    char filename3[60];

    SMatrix C;
    mC = 60000;
    nC = 1024;
    init_SpecialSMatrix(&C, mC, nC);
    for (int k = 0; k < 120; k++)
    {
        gettimeofday(&t3, NULL);
        char filenum[5];
        int k1 = k + 1;
        snprintf(filenum, sizeof(filenum), "%d", k1);

        strcpy(filename3, neuronfile1);
        strcat(filename3, filenum);
        strcat(filename3, neuronfile2);

        SMatrix tmp;
        SMatrix tmp_csc;
        mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename3);
        // printf("%d %d\n",mB,nB);
        // mC = mA;
        // nC = nB;
        tmp.value = (VALUE_TYPE *)malloc((nnzB) * sizeof(VALUE_TYPE));
        tmp.columnindex = (int *)malloc((nnzB) * sizeof(int));
        tmp.rowpointer = (int *)malloc((mB + 1) * sizeof(int));
        tmp_csc.value = (VALUE_TYPE *)malloc((nnzB) * sizeof(VALUE_TYPE));
        tmp_csc.columnindex = (int *)malloc((nnzB) * sizeof(int));
        tmp_csc.rowpointer = (int *)malloc((nB + 1) * sizeof(int));
        mmio_data(tmp.rowpointer, tmp.columnindex, tmp.value, filename3);
        gettimeofday(&t4, NULL);
        time_read_sum += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        double time_read = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        CSR_to_CSC(tmp, &tmp_csc, mB, nB);
        gettimeofday(&t3, NULL);
        time_trans_sum += (t3.tv_sec - t4.tv_sec) * 1000.0 + (t3.tv_usec - t4.tv_usec) / 1000.0;
        double time_trans = (t3.tv_sec - t4.tv_sec) * 1000.0 + (t3.tv_usec - t4.tv_usec) / 1000.0;
        multiply_SpecialSMatrixCSR_with_SMatrixCSC_get_SpecialSMatrixCSR(A, tmp_csc, C, mA, nA, nB, bias);
        copy_SpecialSMtrix(&A, &C, mC, nC);
        gettimeofday(&t4, NULL);
        time_compute_sum += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        double time_compute = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;

        printf("k = %3.d, read time: %6.2f ms, trans time: %6.2f ms, compute time: %7.2f ms\n", k + 1,
               time_read, time_trans, time_compute);

        free(tmp.value);
        free(tmp.columnindex);
        free(tmp.rowpointer);
        free(tmp_csc.value);
        free(tmp_csc.columnindex);
        free(tmp_csc.rowpointer);
    }
    // printf("Weight matrix load time: %f ms \n", time_load);

    gettimeofday(&t2, NULL);
    time_total = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("The read    time cost: %4.2f ms \n", time_read_sum);
    printf("The trans   time cost: %4.2f ms \n", time_trans_sum);
    printf("The compute time cost: %4.2f ms \n", time_compute_sum);
    printf("The whole   time cost: %4.2f ms \n", time_total);

    // double time_inference = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    // printf("Inference time: %f ms \n", time_inference);

    // check results
    printf("test\n");
    FILE *fs;
    fs = fopen("sparse-images-1024-1.tsv", "w+");
    check(fs, A, mA, nA);
    fclose(fs);
    FILE *fp2 = NULL;

    fp2 = fopen("sparse-images-1024-1.tsv", "rb");
    if (fp2 == NULL)
    {
        printf("Error:Open file fail!\n");
    }

    fseek(fp2, 0, SEEK_END);
    size2 = ftell(fp2);
    rewind(fp2);

    tc2 = (int *)malloc(sizeof(int) * size2 / 4);

    int readnum2 = fread(tc2, 4, size2 / 4, fp2);

    fclose(fp2);

    FILE *fp1;

    fp1 = fopen("neuron1024-l120-categories.tsv", "rb");
    if (fp1 == NULL)
    {
        printf("Error:Open file fail!\n");
    }

    fseek(fp1, 0, SEEK_END);
    size1 = ftell(fp1);
    rewind(fp1);

    tc1 = (int *)malloc(sizeof(int) * size1 / 4);

    int readnum1 = fread(tc1, 4, size1 / 4, fp1);

    fclose(fp1);
    int judge = 0;
    for (int i = 0; i < size1 / 4; i++)
    {
        if (tc1[i] - tc2[i] != 0)
        {
            judge++;
        }
    }
    printf("judge:%d\n", judge);
    printf("total:%d\n", size1 / 4);
    if (judge == 0)
    {
        printf("CHALLENGE PASSED\n");
    }
    else
    {
        printf("CHALLENGE FAILED\n");
    }

    return 0;
}
