/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#ifndef __MAX_ENT_MODEL__
#define __MAX_ENT_MODEL__

#include <vector>

#include "dataManager.h"

namespace maxent
{

class MaxEntModel
{

public:
   /**
    *@brief   build the inner data structure from the training file
    *@param   trainFileName : the name of the training file
	*@param	  freq : whether the training file contains the frequency of the event
    *         format of the training file:
    *         (frequency) class_name feature_1 feature_2 ... feature_n
    *@param   select : whether to perform the feature selection function
    *@return  1 to success, 0 to fail
    */
    bool initModel(const char *trainFileName,
                   bool freq = false,
                   bool select = false
                   );
   /**
    *@brief  train the model
    *@param  iter   : times to iterate
    *@param  method : algorithm to train the model
    *@param  tol    : tolerance to stop the training
    *@param  sigma  : parameter of Guass smoothing
    *@param  alpha  : parameter of exponent smoothing
    *@return 1 to success, 0 to fail
    */
    bool trainModel(size_t iter,
                    const std::string & method = "SCGIS",
                    double tol = 1E-03,
                    double sigma2 = 0.0,
                    double alpha = 0.0
                    );
   /**
    *@brief  save the model to a file
    *@param  name of the file to save the model
    *@return 1 to success, 0 to fail
    */
    bool saveModel(const char *modelFileName = "MAXENT.model");
   /**
    *@brief  load the model
    *@param  name of file to load the model
    *@return 1 to success, 0 to fail
    */
    bool loadModel(const char *modelFileName = "MAXENT.model");
   /**
    *@brief  classify a given context to a most probable class
    *@param  className : the predicted. The value of this parameter
                         will be assigned after the call of this function
    *@return the probality
    */
    double predict(std::string & className,
                   const std::vector<std::string> & context);
   /**
    *@brief  the probability of a given context with a specific class
    *@param
    *@return the probability
    */
    double predict(const std::vector<std::string> & context,
                   const std::string & className);
   /**
    *@brief  get the probability of all the class to a certain context
    *@param
    *@return the index of the class with the maxmail probability
    */
    size_t predict(const std::vector<std::string> & context,
                   std::vector<std::pair<std::string, double> > & outcome);

private:
    DataManager mModelInfo;
    MaxEntMap mClassMap;
    MaxEntMap mFetMap;

private:
    void _convert(std::vector<size_t> & fetVec, const std::vector<std::string> & context, const MaxEntMap & mp);

};

}
#endif
