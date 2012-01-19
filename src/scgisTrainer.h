/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#ifndef __SCGIS_TRAINER__
#define __SCGIS_TRAINER__

#include "dataManager.h"
#include "maxEntTrainer.h"

namespace maxent
{

class ScgisTrainer : public MaxEntTrainer
{
public:

	ScgisTrainer(
            DataManager & _modelInfo,
            size_t _iter,
            double _tol,
            double _sigma2,
            double _alpha);

    bool train();

private:

    void _initParams();

private:
    std::vector<std::vector<double> > mS;
    std::vector<double> mZ;
    // featureId : eventId eid ... eid
    std::vector<std::vector<int> > mParam;
};

}

#endif
