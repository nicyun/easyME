/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#include <algorithm>
#include <vector>
#include <utility>

#include <math.h>

#include "dataManager.h"

#define DEBUG 0
#include <fstream>
#include <iostream>

using namespace std;
using namespace maxent;

DataManager::DataManager()
        : mMaxCid(1), mMaxFid(1), mFfCnt(0), mTotEvent(0)
{
	mParamSet.resize(1);
}

void DataManager::clearParam()
{
	mParamSet.clear();
    mParamSet.resize(mMaxFid + 1);
    // always an useless feature
    mMaxFid = 1;
    mFfCnt = 0;
}

int DataManager::getClassPosition(size_t classId, size_t fetId)
{
    int hi = mParamSet[fetId].size() - 1, lo = 0;
    while(hi >= lo){
        int d = (hi + lo) / 2;
        if(mParamSet[fetId][d].first == classId)
            return d;
        if(mParamSet[fetId][d].first < classId)
            lo = d + 1;
        else
            hi = d - 1;
    }
    return -1;
}

size_t DataManager::getAllProbs(
        const context_iterator begin,
        const context_iterator end,
        vector<double> & probs)
{
    probs.clear();
    probs.resize(mMaxCid + 1, 0);
    for(context_iterator cit = begin; cit != end; ++cit){
        size_t fid = *cit;
        for(param_iterator it = getParamBegin(fid),
                pend = getParamEnd(fid);
                it != pend; ++it){
            size_t cid = it->first;
            probs[cid] += it->second;
        }
    }
    size_t maxK = 0;
    double sum = 0;
    for(size_t i = 1; i <= mMaxCid; i++){
        probs[i] = exp(probs[i]);
        sum += probs[i];
        if(probs[i] > probs[maxK])
            maxK = i;
    }
    for(size_t i = 1; i <= mMaxCid; ++i)
        probs[i] /= sum;
    return maxK;
}

void DataManager::getObserves(vector<vector<double> > & observed)
{
	size_t mMaxFid = getFetNum();
	size_t eventNum = getEventNum();
    observed.clear();
    observed.resize(mMaxFid + 1);
    for(size_t i = 1; i <= mMaxFid; ++i){
        DataManager::param_iterator begin = getParamBegin(i);
        DataManager::param_iterator end = getParamEnd(i);
        size_t n = end - begin;
        observed[i].resize(n, 0);
    }

    for(size_t i = 0; i < eventNum; ++i){
        size_t classId = getEventClassId(i);
        size_t count = getEventCount(i);
        for(DataManager::context_iterator it = getContextBegin(i),
                end = getContextEnd(i);
                it != end; ++it){
            int fid = *it;
            int pos = getClassPosition(classId, fid);
            if(pos != -1)
                observed[fid][pos] += count;
        }
    }

#if DEBUG
    ofstream fout("obs.out");
    for(size_t i = 0; i < observed.size(); i++){
        for(size_t j = 0; j < observed[i].size(); j++)
            fout << observed[i][j] << " ";
        fout << endl;
    }
#endif
}

double DataManager::getExpects(vector<vector<double> > & expected)
{
    vector<double> probs;
    expected.clear();
    expected.resize(mParamSet.size());
    for(size_t i = 0; i < mParamSet.size(); ++i)
        expected[i].resize(mParamSet[i].size(), 0);
    double logLike = 0;
    for(size_t i = 0, eventNum = getEventNum(); i < eventNum; ++i){
        context_iterator ctxtBegin = getContextBegin(i);
        context_iterator ctxtEnd = getContextEnd(i);
        getAllProbs(ctxtBegin, ctxtEnd, probs);
        size_t count = getEventCount(i);
        size_t classId = getEventClassId(i);
        vector<double> newProbs;
        for(size_t i = 0; i < probs.size(); ++i)
        	newProbs.push_back(probs[i] * count);
        for(context_iterator it = ctxtBegin; it != ctxtEnd; ++it){
			size_t fid = *it;
			param_iterator pBegin = getParamBegin(fid);
			param_iterator pEnd = getParamEnd(fid);
			for(param_iterator pit = pBegin; pit != pEnd; ++pit){
				int pos = pit - pBegin;
				size_t cid = pit->first;
				expected[fid][pos] += newProbs[cid];
			}
		}
		logLike += log(probs[classId]) * count;
    }
   return logLike;
}

bool mePairEqual(const pair<size_t, double> & a, const pair<size_t, double> & b)
{
    return a.first == b.first;
}
void DataManager::endAddFeature()
{
	mFfCnt = 0;
    for(size_t i = 0; i < mParamSet.size(); ++i){
        sort(mParamSet[i].begin(), mParamSet[i].end());
        size_t size = unique(mParamSet[i].begin(), mParamSet[i].end(), mePairEqual) - mParamSet[i].begin();
        mParamSet[i].resize(size);
        mFfCnt += size;
    }

#if DEBUG
    ofstream fout("param.out");
    for(size_t i = 0; i < mParamSet.size(); i++){
        for(size_t j = 0; j < mParamSet[i].size(); j++)
            fout << mParamSet[i][j].second << " ";
        fout << endl;
    }
#endif
}

void DataManager::getAllFeatures()
{
    size_t eventNum = getEventNum();
    for(size_t i = 0; i < eventNum; ++i){
        int cid = getEventClassId(i);
        for(context_iterator it = getContextBegin(i),
                end = getContextEnd(i);
                it != end; ++it){
            int fid = *it;
            addFeature(cid, fid);
        }
    }
    endAddFeature();

#if DEBUG
    ofstream fout("map.out");
    for(size_t i = 0; i < eventNum; ++i){
        size_t cid = getEventClassId(i);
        size_t count = getEventCount(i);
        fout << count << " " << cid;
        sort(mEventSet[i].context.begin(), mEventSet[i].context.end());
        for(context_iterator it = getContextBegin(i),
                end = getContextEnd(i);
                it != end; ++it){
            int fid = *it;
            fout << " " << fid;
        }
        fout << endl;
    }
#endif
}

void DataManager::processEventSet()
{
    sort(mEventSet.begin(), mEventSet.end());
    vector<Event> newEventSet;
    mEventSet.push_back(Event(0, 0, vector<size_t>()));
    for(size_t i = 0, cnt = 0; i < mEventSet.size() - 1; ++i){
        if(mEventSet[i] == mEventSet[i+1])
            cnt += mEventSet[i].count;
        else{
            mEventSet[i].count += cnt;
            newEventSet.push_back(mEventSet[i]);
            cnt = 0;
        }
    }
    mEventSet = newEventSet;
}

