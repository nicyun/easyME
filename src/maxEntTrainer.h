/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#ifndef __MAX_ENT_TRAINER__
#define __MAX_ENT_TRAINER__

#include "dataManager.h"

namespace maxent{

class MaxEntTrainer
{
public:
	MaxEntTrainer(
            DataManager & _modelInfo,
            size_t _iter,
            double _tol,
            double _sigma2,
            double _alpha);

    virtual ~MaxEntTrainer(){}

public:

    virtual bool train() = 0;

protected:

    double _newton(double f_q, double f_ref, double lambdaNow, double tol = 1.0E-6);

private:

    void _initTrainer(void);

protected:
    DataManager & mModelInfo;
    size_t mIter;
    double mEps;
    double mSigma2;
    double mAlpha;

protected:
    size_t mTotEvent;
    size_t mEventNum;
    size_t mMaxFid;
    size_t mMaxCid;

protected:
    std::vector<std::vector<double> > mObserved;

protected:
    double mSlowF;

};

}

#endif
