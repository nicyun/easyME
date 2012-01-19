/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#ifndef __GIS_TRAINER__
#define __GIS_TRAINER__

#include "dataManager.h"
#include "maxEntTrainer.h"

namespace maxent{

class GisTrainer : public MaxEntTrainer
{
public:

    GisTrainer(
            DataManager & _modelInfo,
            size_t _iter,
            double _tol,
            double _sigma2,
            double _alpha);

    bool train();

private:
    double _getSlowFactor();

};

}

#endif
