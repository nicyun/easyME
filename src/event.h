/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#ifndef __EVENT__
#define __EVENT__

#include <vector>
#include "dataManager.h"

namespace maxent
{

struct Event
{
typedef std::vector<size_t>::iterator context_iterator;
public:
    Event(void):classId(0) { }
    Event(size_t _count, size_t _classId, std::vector<size_t> _context){
        count = _count, classId = _classId, context = _context;
    }
public:
    size_t classId;
    std::vector<size_t> context;
    size_t count;
public:
    // for sort
    bool operator < (const Event & b) const{
        if(classId != b.classId) return classId < b.classId;
        for(size_t i = 0, size = std::min(context.size(), b.context.size()); i < size; ++i)
            if(context[i] != b.context[i])
                return context[i] < b.context[i];
        return context.size() < b.context.size();
    }
    // for unique
    bool operator == (const Event & b) const{
        if(classId != b.classId) return false;
        if(context.size() != b.context.size()) return false;
        for(size_t i = 0; i < context.size(); ++i)
            if(context[i] != b.context[i])
                return false;
        return true;
    }
};

}

#endif
