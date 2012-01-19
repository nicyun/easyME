/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#ifndef __DATA_MANAGER__
#define __DATA_MANAGER__

#include <vector>

#include "event.h"
#include "maxEntMap.h"

namespace maxent
{
class DataManager
{
typedef std::pair<size_t, double> Pair;
typedef std::vector<Pair> Param;

public:

typedef Event::context_iterator context_iterator;
typedef Param::iterator param_iterator;

public:
    DataManager();

private:
    std::vector<Event> mEventSet;
    std::vector<Param> mParamSet;
    size_t mMaxCid;
    size_t mMaxFid;
    // the number of feature function
    size_t mFfCnt;
    size_t mTotEvent;
public:
    inline void addEvent(size_t count, size_t classId, const std::vector<size_t> & fetVec);
    void processEventSet();
    void getAllFeatures();
    inline void addFeature(size_t classId, size_t fetId, double weight = 0.0);
    void endAddFeature();
   	void clearParam();
public:
    inline size_t getEventNum();
    inline size_t getFetNum();
    inline size_t getClassNum();
    inline size_t getAllEventFreq();
    inline size_t getFeatureCount();
    /// for param set
    inline param_iterator getParamBegin(size_t fetId);
    inline param_iterator getParamEnd(size_t fetId);
    inline void setLambda(size_t fetId, size_t classPos, double weight);
    inline void incLambda(size_t fetId, size_t classPos, double incer);
    // -1 for not found
    int getClassPosition(size_t classId, size_t fetId);
    /// for event set
    inline context_iterator getContextBegin(size_t eventId);
    inline context_iterator getContextEnd(size_t eventId);
    inline size_t getEventCount(size_t eventId);
    inline size_t getEventClassId(size_t evenId);
public:
	void getObserves(std::vector<std::vector<double> > & observed);
    double getExpects(std::vector<std::vector<double> > & expects);
    size_t getAllProbs(const context_iterator begin, const context_iterator end, std::vector<double> & probs);
};

size_t DataManager::getEventNum()
{
    return mEventSet.size();
}

size_t DataManager::getFetNum()
{
    return mMaxFid;
}

size_t DataManager::getClassNum()
{
    return mMaxCid;
}

size_t DataManager::getAllEventFreq()
{
	return mTotEvent;
}

size_t DataManager::getFeatureCount()
{
    return mFfCnt;
}

void DataManager::addEvent(size_t count, size_t classId, const std::vector<size_t> & fetVec)
{
	mTotEvent += count;
    mEventSet.push_back(Event(count, classId, fetVec));
}

void DataManager::addFeature(size_t classId, size_t fetId, double weight)
{
    if(mParamSet.size() < fetId + 1) mParamSet.resize(fetId + 1);
    if(classId > mMaxCid) mMaxCid = classId;
    if(fetId > mMaxFid) mMaxFid = fetId;
    mParamSet[fetId].push_back(std::make_pair(classId, weight));
}

DataManager::param_iterator DataManager::getParamBegin(size_t fetId)
{
    return mParamSet[fetId].begin();
}

DataManager::param_iterator DataManager::getParamEnd(size_t fetId)
{
    return mParamSet[fetId].end();
}

void DataManager::setLambda(size_t fetId, size_t classPos, double weight)
{
    mParamSet[fetId][classPos].second = weight;
}

void DataManager::incLambda(size_t fetId, size_t classPos, double incer)
{
    mParamSet[fetId][classPos].second += incer;
}

DataManager::context_iterator DataManager::getContextBegin(size_t eventId)
{
    return mEventSet[eventId].context.begin();
}

DataManager::context_iterator DataManager::getContextEnd(size_t eventId)
{
    return mEventSet[eventId].context.end();
}

size_t DataManager::getEventCount(size_t eventId)
{
    return mEventSet[eventId].count;
}

size_t DataManager::getEventClassId(size_t eventId)
{
    return mEventSet[eventId].classId;
}

}

#endif
