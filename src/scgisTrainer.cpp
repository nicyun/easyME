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

#include "scgisTrainer.h"

using namespace maxent;
using namespace std;

ScgisTrainer::ScgisTrainer(DataManager & _modelInfo, size_t _iter,
		double _tol, double _sigma2, double _alpha)
		: MaxEntTrainer(_modelInfo, _iter, _tol, _sigma2, _alpha)
{
    mSlowF = 1;
    _initParams();
}

void ScgisTrainer::_initParams()
{
    mZ.resize(mEventNum, mMaxCid);
    mS.resize(mMaxCid + 1);
    for(size_t i = 1; i <= mMaxCid; ++i)
        mS[i].resize(mEventNum, 1);
    mParam.resize(mMaxFid + 1);
    for(size_t i = 0; i < mEventNum; ++i){
        for(DataManager::context_iterator it = mModelInfo.getContextBegin(i),
				end = mModelInfo.getContextEnd(i);it != end; ++it){
            size_t fid = *it;
            mParam[fid].push_back(i);
        }
    }
}

bool ScgisTrainer::train()
{
    double preLogLike = -1e10;
    cout << "begin SCGIS training ..." << endl;
    for(size_t it = 0; it < mIter; ++it){
        for(size_t i = 1; i <= mMaxFid; ++i){
			DataManager::param_iterator begin = mModelInfo.getParamBegin(i);
			DataManager::param_iterator end = mModelInfo.getParamEnd(i);
            for(DataManager::param_iterator it = begin; it != end; ++it){
                int j = it - begin;
                size_t cid = it->first;
                double expect = 0;
                for(int k = mParam[i].size() - 1; k >= 0; --k){
                    size_t eid = mParam[i][k];
                    size_t count = mModelInfo.getEventCount(eid);
                    expect += mS[cid][eid] / mZ[eid] * count;
                }
                double inc = 0;
                if(mSigma2){
                    inc = _newton(expect, mObserved[i][j], it->second);
                }
                else if(mAlpha){
                	if(mObserved[i][j] - mAlpha <= 0)
                    	continue;
                    inc = log(mObserved[i][j] - mAlpha) - log(expect);
                	if(inc + it->second <= 0)
                    	inc = -it->second;
                }
                else{
					inc = log(mObserved[i][j]) - log(expect);
				}
                mModelInfo.incLambda(i, j, inc);
                for(int k = mParam[i].size() - 1; k >= 0; --k){
                    size_t eid = mParam[i][k];
                    mZ[eid] -= mS[cid][eid];
                    mS[cid][eid] *= exp(inc);
                    mZ[eid] += mS[cid][eid];
                }
            }
        }
        double newLogLike = 0;
        for(size_t i = 0; i < mEventNum; ++i){
            size_t count = mModelInfo.getEventCount(i);
            size_t cid = mModelInfo.getEventClassId(i);
            newLogLike += (log(mS[cid][i]) - log(mZ[i])) * count;
        }

        cout << "Iter = " << it + 1 << "    Loglike = " << newLogLike << endl;

        if(fabs((newLogLike - preLogLike) / preLogLike) < mEps)
            break;
        preLogLike = newLogLike;
    }
    return true;
}
