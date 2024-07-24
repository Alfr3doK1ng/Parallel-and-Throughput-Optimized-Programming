#include "ConjugateGradients.h"
#include "ConjugateGradientsNew.h"
#include "Timer.h"
#include "Utilities.h"

Timer timerLaplacian;
Timer timerInnerProduct;
Timer timerNorm;
Timer timerCopy;
Timer timerSaxpy;
Timer timerCombinedKernel;
Timer timerLaplacianAndInnerProduct;
Timer timerCombinedSaxpyNorm;

int main(int argc, char *argv[])
{
    using array_t = float (&) [XDIM][YDIM][ZDIM];

    float *xRaw = new float [XDIM*YDIM*ZDIM];
    float *fRaw = new float [XDIM*YDIM*ZDIM];
    float *pRaw = new float [XDIM*YDIM*ZDIM];
    float *rRaw = new float [XDIM*YDIM*ZDIM];
    float *zRaw = new float [XDIM*YDIM*ZDIM];
    
    array_t x = reinterpret_cast<array_t>(*xRaw);
    array_t f = reinterpret_cast<array_t>(*fRaw);
    array_t p = reinterpret_cast<array_t>(*pRaw);
    array_t r = reinterpret_cast<array_t>(*rRaw);
    array_t z = reinterpret_cast<array_t>(*zRaw);
    
    // Initialization
    {
        Timer timer;
        timer.Start();
        InitializeProblem(x, f);
        timer.Stop("Initialization : ");
    }

    // Call Conjugate Gradients algorithm
    Timer timerMain;
    timerMain.Start();

    timerLaplacian.Reset();
    timerInnerProduct.Reset();
    timerNorm.Reset();
    timerCopy.Reset();
    timerSaxpy.Reset();
    timerCombinedKernel.Reset();
    timerLaplacianAndInnerProduct.Reset();
    timerCombinedSaxpyNorm.Reset();

    // ConjugateGradients(x, f, p, r, z, false);
    ConjugateGradientsNew(x, f, p, r, z, false);

    timerLaplacian.Print("Total Laplacian Time : ");
    timerInnerProduct.Print("Total InnerProduct Time : ");
    timerNorm.Print("Total Norm Time : ");
    timerCopy.Print("Total Copy Time : ");
    timerSaxpy.Print("Total Saxpy Time : ");
    timerCombinedKernel.Print("Total CombinedKernel Time : ");
    timerLaplacianAndInnerProduct.Print("Total LaplacianAndInnerProduct Time : ");
    timerCombinedSaxpyNorm.Print("Total CombinedSaxpyNorm Time : ");

    timerMain.Stop("Main : ");

    return 0;
}
