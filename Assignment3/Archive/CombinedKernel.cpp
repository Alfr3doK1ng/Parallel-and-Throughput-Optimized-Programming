#include "CombinedKernel.h"
#include <algorithm>
#include <cmath>     
#include <omp.h>


float CombinedKernel(
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM])
{
    float maxVal = 0.0; // For norm calculation

#pragma omp parallel for reduction(max:maxVal)
    for (int i = 1; i < XDIM-1; i++) {
        for (int j = 1; j < YDIM-1; j++) {
            for (int k = 1; k < ZDIM-1; k++) {
                // Compute Laplacian
                float Lu = -6 * x[i][j][k]
                           + x[i+1][j][k] + x[i-1][j][k]
                           + x[i][j+1][k] + x[i][j-1][k]
                           + x[i][j][k+1] + x[i][j][k-1];
                // Perform Saxpy (note that Lu is used instead of a separate z array)
                r[i][j][k] = Lu * (-1) + f[i][j][k]; // Assuming scale is -1 as in your example

                // Update maxVal for Norm calculation
                // Note: This is not the final value yet, just the max of this operation
                maxVal = std::max(maxVal, std::abs(r[i][j][k]));
            }
        }
    }

    // maxVal now holds the maximum absolute value encountered during the operation,
    // which corresponds to the Norm of the resultant array r.
    return maxVal;
}

float ComputeLaplacianAndInnerProduct(const float (&u)[XDIM][YDIM][ZDIM], float (&z)[XDIM][YDIM][ZDIM])
{
    double innerProductResult = 0.0;

#pragma omp parallel for reduction(+:innerProductResult) collapse(3)
    for (int i = 1; i < XDIM-1; i++) {
        for (int j = 1; j < YDIM-1; j++) {
            for (int k = 1; k < ZDIM-1; k++) {
                // Directly compute the Laplacian of u[i][j][k] and use it in the inner product calculation
                z[i][j][k] = -6.0f * u[i][j][k] 
                                + u[i+1][j][k] 
                                + u[i-1][j][k] 
                                + u[i][j+1][k] 
                                + u[i][j-1][k] 
                                + u[i][j][k+1] 
                                + u[i][j][k-1];

                // Accumulate the result for the inner product of the on-the-fly Laplacian and z
                innerProductResult += (double)u[i][j][k] * (double)z[i][j][k];
            }
        }
    }

    return (float) innerProductResult;
}

float CombinedSaxpyNorm(
    const float (&m) [XDIM][YDIM][ZDIM],
    const float (&x)[XDIM][YDIM][ZDIM], 
    const float (&y)[XDIM][YDIM][ZDIM], 
    float (&z)[XDIM][YDIM][ZDIM],
    const float scale) {
    float result = 0.;
    #pragma omp parallel for reduction(max:result)
        for (int i = 1; i < XDIM-1; i++)
        for (int j = 1; j < YDIM-1; j++)
        for (int k = 1; k < ZDIM-1; k++) {
            z[i][j][k] = x[i][j][k] * scale + y[i][j][k];
            result = std::max(result,std::abs(m[i][j][k]));}
        return result;
}