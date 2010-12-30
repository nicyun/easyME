#include <iostream>

#include <stdlib.h>
#include <math.h>

#include "lbfgsTrainer.h"

using namespace std;
using namespace maxent;

LbfgsTrainer::LbfgsTrainer(DataManager & _modelInfo, size_t _iter,
		double _tol, double _sigma2, double _alpha)
		: MaxEntTrainer(_modelInfo, _iter, _tol, _sigma2, _alpha)
{
	mFfN = mModelInfo.getFeatureCount();
	mX = lbfgs_malloc(mFfN);
	if(!mX){
		cerr << "ERROR LBFGS : "
			 << "Failed to allocate a memory block for variables."
			 << endl;
		exit(-1);
	}
	_initLbfgsParam();
}

LbfgsTrainer::~LbfgsTrainer()
{
	lbfgs_free(mX);
}

void LbfgsTrainer::_initLbfgsParam()
{
	lbfgs_parameter_init(&mLbfgsParam);
	mLbfgsParam.m = 5;
	mLbfgsParam.epsilon = mEps;
	mLbfgsParam.max_iterations = mIter;
}

int LbfgsTrainer::progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int nit,
        int ls)
{
    cout << "Iter = " << nit << "	Loglike = " << fx << endl;
    return 0;
}

void LbfgsTrainer::_updateLambda()
{
	for(size_t i = 1, k = 0; i <= mMaxFid; ++i)
        for(size_t j = 0; j < mObserved[i].size(); ++j){
        	mModelInfo.setLambda(i, j, mX[k++]);
    	}
}

lbfgsfloatval_t LbfgsTrainer::evaluate(
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *grad,
        const int n,
        const lbfgsfloatval_t step)
{
	_updateLambda();
	vector<vector<double> > expected;
    lbfgsfloatval_t fVal = -mModelInfo.getExpects(expected);
	for(size_t i = 1, k = 0; i <= mMaxFid; ++i)
		for(size_t j = 0; j < mObserved[i].size(); ++j)
			grad[k++] = expected[i][j] - mObserved[i][j];
	if (mSigma2) {
    	for(size_t i = 0; i < mFfN; ++i){
       		double penality = x[i] / mSigma2;
        	grad[i] += penality;
           	fVal += (penality * x[i]) / 2;
        }
    }
    else if(mAlpha){
		cerr << "ERROR LBFGS: "
			 << "exponential penality is not supported currently!"
			 << endl;
		exit(-1);
    }
	return fVal;
}

bool LbfgsTrainer::train()
{
	fill(mX, mX + mFfN, 0.0);
    cout << "begin LBFGS training ..." << endl;
    int ret = lbfgs(mFfN, mX, NULL, _evaluate, _progress, this, &mLbfgsParam);
    _updateLambda();
    if(ret){
    	switch (ret){
    		case -997:
    			cout << "LBFGS finished with max mIter reached!" << endl;
    			break;
    		default:
    			cerr << "ERROR : "
					 << "LBFGS exit with error code "
					 << ret << endl;
    			exit(-1);
    	}
    }
	return true;
}
