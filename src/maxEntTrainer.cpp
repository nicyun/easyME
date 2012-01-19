/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#include <iostream>
#include <math.h>
#include "maxEntTrainer.h"

namespace maxent{

using namespace std;

MaxEntTrainer::MaxEntTrainer(DataManager & _modelInfo, size_t _iter,
        double _tol, double _sigma2, double _alpha)
        : mModelInfo(_modelInfo), mIter(_iter), mEps(_tol), mSigma2(_sigma2), mAlpha(_alpha)
{
    _initTrainer();
}

void MaxEntTrainer::_initTrainer(void)
{
    mEventNum = mModelInfo.getEventNum();
    mMaxFid = mModelInfo.getFetNum();
    mMaxCid = mModelInfo.getClassNum();
    mTotEvent = mModelInfo.getAllEventFreq();
    mModelInfo.getObserves(mObserved);
}

// Calculate the ith GIS parameter updates with Gaussian prior
// using Newton-Raphson method
// the update rule is the solution of the following equation:
//                                   lambda_i + delta_i
// E_ref = E_q * exp (C * delta_i) + ------------------ * N
//                                       sigma_i^2
// note: E_ref and E_q were not divided by N
// this function is copied from Le Zhang's work
double MaxEntTrainer::_newton(double f_q, double f_ref, double lambdaNow, double mEps)
{
    size_t maxiter = 50;
    double x0 = 0.0;
    double x = 0.0;

    for (size_t mIter = 1; mIter <= maxiter; ++mIter) {
        double t = f_q * exp(mSlowF * x0);
        double fval = t + mTotEvent * (lambdaNow + x0) / mSigma2 - f_ref;
        double fpval = t * mSlowF + mTotEvent / mSigma2;
        if (fpval == 0) {
            cerr << "Warning: zero-derivative encountered in newton() method." << endl;
            return x0;
        }
        x = x0 - fval/fpval;
        if (fabs(x-x0) < mEps)
            return x;
        x0 = x;
    }
    cerr << "Failed to converge after 50 iterations in newton() method" << endl;
    exit(-1);
}

}
