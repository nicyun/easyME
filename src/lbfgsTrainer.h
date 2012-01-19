/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#ifndef __LBFGS_TRAINER__
#define __LBFGS_TRAINER__

#include "dataManager.h"
#include "maxEntTrainer.h"
#include "lbfgs.h"

namespace maxent{

class LbfgsTrainer : public MaxEntTrainer
{
public:

    LbfgsTrainer(
            DataManager & _modelInfo,
            size_t _iter,
            double _tol,
            double _sigma2,
            double _alpha);

    ~LbfgsTrainer();

public:

    bool train();

private:

	lbfgsfloatval_t evaluate(
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        );

	int progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        );

    void _initLbfgsParam();
    void _updateLambda();

    static lbfgsfloatval_t _evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step)
    {
        return reinterpret_cast<LbfgsTrainer*>(instance)->evaluate(x, g, n, step);
    }

    static int _progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls)
    {
        return reinterpret_cast<LbfgsTrainer*>(instance)->
        			progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

private:
	size_t mFfN;
	lbfgsfloatval_t * mX;
	lbfgs_parameter_t mLbfgsParam;
};

}

#endif
