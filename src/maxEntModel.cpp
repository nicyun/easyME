/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#include <fstream>
#include <iostream>
#include <sstream>

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "maxEntModel.h"
#include "maxEntTrainer.h"
#include "gisTrainer.h"
#include "scgisTrainer.h"
#include "lbfgsTrainer.h"

using namespace std;
using namespace maxent;

bool MaxEntModel::initModel(
        const char * trainFileName,
        bool freq,
        bool isSelectFeature)
{
    cout << "begin Initialization..." << endl;
    ifstream fin(trainFileName);
    if(!fin){
        cerr << "train file not exit!" << endl;
        exit(-1);
    }
    string line, str;
    vector<size_t> context;
    size_t count = 1;
    // each line is a event, it looks like this:
    // (count) className fetName ... fetName
    while(getline(fin, line)){
        istringstream sin(line);
        // with freq ?
        if(freq) sin >> count;
        sin >> str;
        size_t classId = mClassMap.insertString(str);
        context.clear();
        while(sin >> str){
            size_t fetId = mFetMap.insertString(str);
            context.push_back(fetId);
        }
        mModelInfo.addEvent(count, classId, context);
    }
    if(!freq) mModelInfo.processEventSet();
    mModelInfo.getAllFeatures();
    cout << "Initialization successful." << endl;
    cout << "num of train events: " << mModelInfo.getEventNum() << endl;
    cout << "max fetId is: " << mModelInfo.getFetNum() << endl;
    cout << "max classId is: " << mModelInfo.getClassNum() << endl;
    cout << "sum of all events' count: " << mModelInfo.getAllEventFreq() << endl;
    cout << "number of feature count is: " << mModelInfo.getFeatureCount() << endl;
    if(isSelectFeature){
        cout << "begin feature selection ..." << endl;
        // do feature select
        // some code...
        cout << "feature selection finished sucessfully" << endl;
        cout << "reduced feature count to " << mModelInfo.getFeatureCount() << endl;
    }
    return true;
}

bool MaxEntModel::trainModel(
        size_t iter,
        const string & method,
        double tol,
        double sigma2,
        double alpha)
{
    MaxEntTrainer *trainer;
    if(method == "GIS")
        trainer = new GisTrainer(mModelInfo, iter, tol, sigma2, alpha);
    else if(method == "SCGIS")
        trainer = new ScgisTrainer(mModelInfo, iter, tol, sigma2, alpha);
    else if(method == "LBFGS")
    	trainer = new LbfgsTrainer(mModelInfo, iter, tol, sigma2, alpha);
    else{
        cerr << "method must be GIS or SCGIS or LBFGS" << endl;
        exit(-1);
    }

    cout << "begin trainning..." << endl;
    trainer->train();
    cout << "Model Training successful." << endl;
    delete trainer;
    return true;
}

bool MaxEntModel::loadModel(const char * modelFileName)
{
    ifstream fin(modelFileName);
    if(!fin){
        cerr << "open model file failed!" << endl;
        exit(-1);
    }
    cout << "load model from " << modelFileName << " ..." << endl;
    string line, str;
    double weight;
    vector<size_t> classVec;
    vector<double> weights;
    mModelInfo.clearParam();
    // each line is a param vector, it may look like this:
    // fetName className weight ... className weight
    while(getline(fin, line)){
        istringstream sin(line);
        sin >> str;
        size_t fetId = mFetMap.insertString(str);
        classVec.clear();
        weights.clear();
        while(sin >> str){
            sin >> weight;
            size_t classId = mClassMap.insertString(str);
            mModelInfo.addFeature(classId, fetId, weight);
        }
    }
    mModelInfo.endAddFeature();
    cout << "model loaded ok!" << endl;
    cout << "num of fets: " << mModelInfo.getFetNum() << endl;
    cout << "num of classes: " << mModelInfo.getClassNum() << endl;
    cout << "tot feature count is: " << mModelInfo.getFeatureCount() << endl;
    return true;
}

bool MaxEntModel::saveModel(const char * modelFileName)
{
    ofstream fout(modelFileName);
    if(!fout){
        cerr << "can't create model file!" << endl;
        exit(-1);
    }

    cout << "save model to " << modelFileName << " ..." << endl;

    size_t fetNum = mModelInfo.getFetNum();
    for(size_t fid = 1; fid <= fetNum; ++fid){
        string fetName, className;
        mFetMap.num2str(fid, fetName);
        ostringstream sout;
        sout << fetName;
        bool empty = 1;
        for(DataManager::param_iterator it = mModelInfo.getParamBegin(fid),
                end = mModelInfo.getParamEnd(fid);
                it != end; ++it){
            if(it->second == 0)
                continue;
            empty = 0;
            mClassMap.num2str(it->first, className);
            sout << " " << className;
            sout << " " << it->second;
        }
        if(!empty) fout << sout.str() << endl;
    }
    cout << "model saved ok!" << endl;
    return true;
}

void MaxEntModel::_convert(vector<size_t> & fetVec, const vector<string> & context, const MaxEntMap & mp)
{
	fetVec.clear();
	for(size_t i = 0; i < context.size(); ++i)
		fetVec.push_back(mp.str2num(context[i]));
}

double MaxEntModel::predict(string & className, const vector<std::string> & context)
{
    vector<size_t> fetVec;
    vector<double> probs;

    _convert(fetVec, context, mFetMap);
	size_t best = mModelInfo.getAllProbs(fetVec.begin(), fetVec.end(), probs);

    mClassMap.num2str(best, className);

    return probs[best];
}

double MaxEntModel::predict(const vector<string> & context, const string & strClass)
{
	vector<size_t> fetVec;
    vector<double> probs;

    _convert(fetVec, context, mFetMap);
    mModelInfo.getAllProbs(fetVec.begin(), fetVec.end(), probs);

    size_t classId = mClassMap.str2num(strClass);

    return probs[classId];
}

size_t MaxEntModel::predict(const vector<string> & context, vector<pair<string, double> > & outcome)
{
	vector<size_t> fetVec;
    vector<double> probs;

    _convert(fetVec, context, mFetMap);
    size_t best = mModelInfo.getAllProbs(fetVec.begin(), fetVec.end(), probs);

    static vector<string> className;
	if(className.size() == 0){
		string tmp;
		for(size_t i = 0; i < probs.size(); ++i){
			mClassMap.num2str(i, tmp);
			className.push_back(tmp);
		}
	}

	outcome.clear();
    for(size_t i = 0; i < probs.size(); ++i)
    	outcome.push_back(make_pair(className[i], probs[i]));

    return best;
}
