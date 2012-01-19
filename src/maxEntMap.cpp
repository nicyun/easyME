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
#include <string>
#include <ext/hash_map>
#include <iostream>

#include "maxEntMap.h"

using namespace std;
using namespace maxent;

MaxEntMap::~MaxEntMap()
{
    // none
}

MaxEntMap::MaxEntMap() : mNumber(1) { }

size_t MaxEntMap::insertString(const string & str)
{
    if(mStr2Num.count(str))
        return mStr2Num[str];
    else {
        mStr2Num[str] = mNumber;
        if(mNum2Str.size() < mNumber + 1)
            mNum2Str.resize(mNumber + 1);
        mNum2Str[mNumber] = str;
        return mNumber++;
    }
}

size_t MaxEntMap::str2num(const string & str) const
{
    __gnu_cxx::hash_map<string, size_t, string_hash>::const_iterator iter = mStr2Num.find(str);
	if(iter != mStr2Num.end())
        return iter -> second;
    else
        return 0;
}

string & MaxEntMap::num2str(const size_t id, string &str) const
{
    if(id < mNum2Str.size())
        str = mNum2Str[id];
    else str = "";
    return str;
}
