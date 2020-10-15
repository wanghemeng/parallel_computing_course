/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

int dense2csr(float *A, int mA, int nA, int nnzA,
              int **hA_csrOffsets, int **hA_columns, float **hA_values)
{
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;
    int *dNnzPerRowA;
    CHECK(cudaMalloc((void **)&dNnzPerRowA, sizeof(int) * mA));
    float *dA;
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * mA * nA));
    CHECK(cudaMemcpy(dA, A, sizeof(float) * mA * nA, cudaMemcpyHostToDevice));
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, mA, nA, descr, dA,
                                mA, dNnzPerRowA, &nnzA));
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * nnzA));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (mA + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * nnzA));
    CHECK_CUSPARSE(cusparseSdense2csr(handle, mA, nA, descr, dA, mA, dNnzPerRowA,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));
    *hA_csrOffsets = (int *)malloc((sizeof(int) * (mA + 1)));
    *hA_columns = (int *)malloc((sizeof(int) * nnzA));
    *hA_values = (int *)malloc((sizeof(float) * nnzA));
    CHECK_CUDA(cudaMemcpy(*hA_csrOffsets, dCsrRowPtrA,
                          (sizeof(int) * (mA + 1)),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(*hA_columns, dCsrColIndA,
                          (sizeof(int) * nnzA),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(*hA_values, dCsrValA,
                          (sizeof(float) * nnzA),
                          cudaMemcpyDeviceToHost));
    return 0;
}

int cuda_csr_csr_csr(int *hA_csrOffsets, int *hA_columns, float *hA_values, int mA, int nA, int nnzA,
                     int *hB_csrOffsets, int *hB_columns, float *hB_values, int mB, int nB, int nnzB,
                     int **hC_csrOffsets, int **hC_columns, float **hC_values)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device memory management: Allocate and copy A, B

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t descr = 0;
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    float *dCsrValA, *dCsrValB, *dCsrValC;
    int *dCsrRowPtrA, *dCsrRowPtrB, *dCsrRowPtrC;
    int *dCsrColIndA, *dCsrColIndB, *dCsrColIndC;
    // copy A
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * nnzA));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (mA + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * nnzA));
    CHECK_CUDA(cudaMemcpy(dCsrRowPtrA, hA_csrOffsets,
                          (mA + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dCsrColIndA, hA_columns, nnzA * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dCsrValA, hA_values,
                          nnzA * sizeof(float), cudaMemcpyHostToDevice))
    // copy B
    CHECK(cudaMalloc((void **)&dCsrValB, sizeof(float) * nnzB));
    CHECK(cudaMalloc((void **)&dCsrRowPtrB, sizeof(int) * (mB + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndB, sizeof(int) * nnzB));
    CHECK_CUDA(cudaMemcpy(dCsrRowPtrB, hB_csrOffsets,
                          (mB + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dCsrColIndB, hB_columns, nnzB * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dCsrValB, hB_values,
                          nnzB * sizeof(float), cudaMemcpyHostToDevice))

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, mA, nA, nnzA,
                                     dCsrRowPtrA, dCsrColIndA, dCsrValA,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, mB, nB, nnzB,
                                     dCsrRowPtrB, dCsrColIndB, dCsrValB,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, mA, nB, 0,
                                     NULL, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer1, bufferSize1));
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1));

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer2, bufferSize2));

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &bufferSize2, dBuffer2));
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                        &C_num_nnz1));
    // allocate matrix C
    CHECK_CUDA(cudaMalloc((void **)&dCsrColIndC, C_num_nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dCsrValC, C_num_nnz1 * sizeof(float)));
    // allocate C offsets
    CHECK_CUDA(cudaMalloc((void **)&dCsrRowPtrC, (mA + 1) * sizeof(int)));
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dCsrRowPtrC, dCsrColIndC, dCsrValC));

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    //--------------------------------------------------------------------------
    // device result copy
    *hC_csrOffsets = (int *)malloc((mA + 1) * sizeof(int));
    *hC_columns = (int *)malloc((C_num_nnz1 * sizeof(int));
    *hC_values = (int *)malloc((C_num_nnz1 * sizeof(float));
    CHECK_CUDA( cudaMemcpy(*hC_csrOffsets, dCsrRowPtrC,
                           (mA + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(*hC_columns, dCsrColIndC,
                           C_num_nnz1 * sizeof(int),
                           cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(*hC_values, dCsrValC,
                           C_num_nnz1 * sizeof(float),
                           cudaMemcpyDeviceToHost) );

    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer1) );
    CHECK_CUDA( cudaFree(dBuffer2) );
    CHECK_CUDA( cudaFree(dCsrRowPtrA) );
    CHECK_CUDA( cudaFree(dCsrColIndA) );
    CHECK_CUDA( cudaFree(dCsrValA) );
    CHECK_CUDA( cudaFree(dCsrRowPtrB) );
    CHECK_CUDA( cudaFree(dCsrColIndB) );
    CHECK_CUDA( cudaFree(dCsrValB) );
    CHECK_CUDA( cudaFree(dCsrRowPtrC) );
    CHECK_CUDA( cudaFree(dCsrColIndC) );
    CHECK_CUDA( cudaFree(dCsrValC) );
    return EXIT_SUCCESS;
}

int cuda_dense_dense_csr(float *A, int mA, int nA, int nnzA,
                         float *B, int mB, int nB, int nnzB,
                         int **hC_csrOffsets, int **hC_columns, float **hC_values)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device memory management: Allocate and copy A, B

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t descr = 0;
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    float *dCsrValA, *dCsrValB, *dCsrValC;
    int *dCsrRowPtrA, *dCsrRowPtrB, *dCsrRowPtrC;
    int *dCsrColIndA, *dCsrColIndB, *dCsrColIndC;

    int *dNnzPerRowA;
    CHECK(cudaMalloc((void **)&dNnzPerRowA, sizeof(int) * mA));
    float *dA;
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * mA * nA));
    CHECK(cudaMemcpy(dA, A, sizeof(float) * mA * nA, cudaMemcpyHostToDevice));
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, mA, nA, descr, dA,
                                mA, dNnzPerRowA, &nnzA));
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * nnzA));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (mA + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * nnzA));
    CHECK_CUSPARSE(cusparseSdense2csr(handle, mA, nA, descr, dA, mA, dNnzPerRowA,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));
    int *dNnzPerRowB;
    CHECK(cudaMalloc((void **)&dNnzPerRowB, sizeof(int) * mB));

    float *dB;
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * mB * nB));
    CHECK(cudaMemcpy(dB, B, sizeof(float) * mB * nB, cudaMemcpyHostToDevice));
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, mB, nB, descr, dB,
                                mB, dNnzPerRowB, &nnzB));
    CHECK(cudaMalloc((void **)&dCsrValB, sizeof(float) * nnzB));
    CHECK(cudaMalloc((void **)&dCsrRowPtrB, sizeof(int) * (mB + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndB, sizeof(int) * nnzB));
    CHECK_CUSPARSE(cusparseSdense2csr(handle, mB, nB, descr, dB, mB, dNnzPerRowB,
                                      dCsrValB, dCsrRowPtrB, dCsrColIndB));

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, mA, nA, nnzA,
                                     dCsrRowPtrA, dCsrColIndA, dCsrValA,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, mB, nB, nnzB,
                                     dCsrRowPtrB, dCsrColIndB, dCsrValB,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, mA, nB, 0,
                                     NULL, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer1, bufferSize1));
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1));

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer2, bufferSize2));

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &bufferSize2, dBuffer2));
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                        &C_num_nnz1));
    // allocate matrix C
    CHECK_CUDA(cudaMalloc((void **)&dCsrColIndC, C_num_nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dCsrValC, C_num_nnz1 * sizeof(float)));
    // allocate C offsets
    CHECK_CUDA(cudaMalloc((void **)&dCsrRowPtrC, (mA + 1) * sizeof(int)));
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dCsrRowPtrC, dCsrColIndC, dCsrValC));

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    //--------------------------------------------------------------------------
    // device result copy
    *hC_csrOffsets = (int *)malloc((mA + 1) * sizeof(int));
    *hC_columns = (int *)malloc((C_num_nnz1 * sizeof(int));
    *hC_values = (int *)malloc((C_num_nnz1 * sizeof(float));
    CHECK_CUDA( cudaMemcpy(*hC_csrOffsets, dCsrRowPtrC,
                           (mA + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(*hC_columns, dCsrColIndC,
                           C_num_nnz1 * sizeof(int),
                           cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(*hC_values, dCsrValC,
                           C_num_nnz1 * sizeof(float),
                           cudaMemcpyDeviceToHost) );

    // device memory deallocation
    CHECK_CUDA( cudaFree(dNnzPerRowA) );
    CHECK_CUDA( cudaFree(dNnzPerRowB) );
    CHECK_CUDA( cudaFree(dBuffer1) );
    CHECK_CUDA( cudaFree(dBuffer2) );
    CHECK_CUDA( cudaFree(dA) );
    CHECK_CUDA( cudaFree(dB) );
    CHECK_CUDA( cudaFree(dCsrRowPtrA) );
    CHECK_CUDA( cudaFree(dCsrColIndA) );
    CHECK_CUDA( cudaFree(dCsrValA) );
    CHECK_CUDA( cudaFree(dCsrRowPtrB) );
    CHECK_CUDA( cudaFree(dCsrColIndB) );
    CHECK_CUDA( cudaFree(dCsrValB) );
    CHECK_CUDA( cudaFree(dCsrRowPtrC) );
    CHECK_CUDA( cudaFree(dCsrColIndC) );
    CHECK_CUDA( cudaFree(dCsrValC) );
    return EXIT_SUCCESS;
}

int cuda_neron_120(float *A, int mA, int nA, int nnzA, float *B, int mB, int nB, int nnzB）
{
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device memory management: Allocate and copy A, B

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t descr = 0;
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    float *dCsrValA, *dCsrValB, *dCsrValC;
    int *dCsrRowPtrA, *dCsrRowPtrB, *dCsrRowPtrC;
    int *dCsrColIndA, *dCsrColIndB, *dCsrColIndC;

    int *dNnzPerRowA;
    CHECK(cudaMalloc((void **)&dNnzPerRowA, sizeof(int) * mA));
    float *dA;
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * mA * nA));
    CHECK(cudaMemcpy(dA, A, sizeof(float) * mA * nA, cudaMemcpyHostToDevice));
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, mA, nA, descr, dA,
                                mA, dNnzPerRowA, &nnzA));
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * nnzA));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (mA + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * nnzA));
    CHECK_CUSPARSE(cusparseSdense2csr(handle, mA, nA, descr, dA, mA, dNnzPerRowA,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));
    int *dNnzPerRowB;
    CHECK(cudaMalloc((void **)&dNnzPerRowB, sizeof(int) * mB));

    float *dB;
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * mB * nB));
    CHECK(cudaMemcpy(dB, B, sizeof(float) * mB * nB, cudaMemcpyHostToDevice));
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, mB, nB, descr, dB,
                                mB, dNnzPerRowB, &nnzB));
    CHECK(cudaMalloc((void **)&dCsrValB, sizeof(float) * nnzB));
    CHECK(cudaMalloc((void **)&dCsrRowPtrB, sizeof(int) * (mB + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndB, sizeof(int) * nnzB));
    CHECK_CUSPARSE(cusparseSdense2csr(handle, mB, nB, descr, dB, mB, dNnzPerRowB,
                                      dCsrValB, dCsrRowPtrB, dCsrColIndB));

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, mA, nA, nnzA,
                                     dCsrRowPtrA, dCsrColIndA, dCsrValA,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, mB, nB, nnzB,
                                     dCsrRowPtrB, dCsrColIndB, dCsrValB,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, mA, nB, 0,
                                     NULL, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer1, bufferSize1));
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1));

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer2, bufferSize2));

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &bufferSize2, dBuffer2));
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                        &C_num_nnz1));
    // allocate matrix C
    CHECK_CUDA(cudaMalloc((void **)&dCsrColIndC, C_num_nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dCsrValC, C_num_nnz1 * sizeof(float)));
    // allocate C offsets
    CHECK_CUDA(cudaMalloc((void **)&dCsrRowPtrC, (mA + 1) * sizeof(int)));
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dCsrRowPtrC, dCsrColIndC, dCsrValC));

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    CHECK_CUDA(cudaFree(dBuffer1));
    CHECK_CUDA(cudaFree(dBuffer2));
    CHECK_CUDA(cudaFree(dCsrColIndA));
    CHECK_CUDA(cudaFree(dCsrValA));
    CHECK_CUDA(cudaFree(dCsrRowPtrA));

    // SpGEMM Computation C * B[k+1] = A
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer1, bufferSize1));
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1));

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer2, bufferSize2));

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &bufferSize2, dBuffer2));
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                        &C_num_nnz1));
    // allocate matrix C
    CHECK_CUDA(cudaMalloc((void **)&dCsrColIndC, C_num_nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dCsrValC, C_num_nnz1 * sizeof(float)));
    // allocate C offsets
    CHECK_CUDA(cudaMalloc((void **)&dCsrRowPtrC, (mA + 1) * sizeof(int)));
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dCsrRowPtrC, dCsrColIndC, dCsrValC));

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));


    return EXIT_SUCCESS;
}