#pragma once

#include "Parameters.h"

float CombinedKernel(
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM]);

float ComputeLaplacianAndInnerProduct(
    const float (&u)[XDIM][YDIM][ZDIM], 
    float (&z)[XDIM][YDIM][ZDIM]);

float CombinedSaxpyNorm(
    const float (&m) [XDIM][YDIM][ZDIM],
    const float (&x)[XDIM][YDIM][ZDIM], 
    const float (&y)[XDIM][YDIM][ZDIM], 
    float (&z)[XDIM][YDIM][ZDIM],
    const float scale);