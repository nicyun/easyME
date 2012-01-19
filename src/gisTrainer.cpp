/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#include <vector>
#include <iostream>

#include <math.h>

#include "gisTrainer.h"

using namespace std;
using namespace maxent;


GisTrainer::GisTrainer(DataManager & _modelInfo, size_t _iter,
		double _tol, double _sigma2, double _alpha)
		: MaxEntTrainer(_modelInfo, _iter, _tol, _sigma2, _alpha)
{
    _getSlowFactor();
    cout << "slow factor is : " << mSlowF << endl;
}

double GisTrainer::_getSlowFactor()
{
    mSlowF = 0;
    for(size_t i = 0; i < mEventNum; ++i){
        double tmpF = 0;
        size_t cid = mModelInfo.getEventClassId(i);
        for(DataManager::context_iterator it = mModelInfo.getContextBegin(i),
				end = mModelInfo.getContextEnd(i);
				it != end; ++it){
            if(mModelInfo.getClassPosition(cid, *it) != -1)
                tmpF += 1;
        }
        if(tmpF > mSlowF)
            mSlowF = tmpF;
    }
    return mSlowF;
}

bool GisTrainer::train()
{
    vector<vector<double> > expected;

    double preLogLike = -1e10;
    std::cout << "begin GIS training ..." << std::endl;
    for(size_t it = 0; it < mIter; ++it){
        double newLogLike = mModelInfo.getExpects(expected);
        for(size_t i = 1; i <= mMaxFid; ++i){
            DataManager::param_iterator begin = mModelInfo.getParamBegin(i);
            DataManager::param_iterator end = mModelInfo.getParamEnd(i);
            for(DataManager::param_iterator it = begin; it != end; ++it)
            {
                size_t j = it - begin;
                double inc = 0;
                if(mSigma2){
                    inc = _newton(expected[i][j], mObserved[i][j], it->second);
                }
                else if(mAlpha){
                	if(mObserved[i][j] - mAlpha <= 0)
                    	continue;
                    inc = (log(mObserved[i][j] - mAlpha) - log(expected[i][j])) / mSlowF;
                	if(it->second + inc <= 0)
                    	inc = -it->second;
                }
                else {
					inc = (log(mObserved[i][j]) - log(expected[i][j])) / mSlowF;
				}
                mModelInfo.incLambda(i, j, inc);
            }
        }

        cout << "Iter = " << it + 1 << "    Loglike = " << newLogLike << endl;

        if(fabs((newLogLike - preLogLike) / preLogLike) < mEps)
            break;
        preLogLike = newLogLike;
    }
    return true;
}
