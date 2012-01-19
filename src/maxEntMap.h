/**
 *  easyME -- a Maximum Entropy toolkit
 *
 *  Copyright(C) 2009
 *
 *  Li Yun <nicyun@gmail.com>
 *  Chen Zhijie <ruoyu928@126.com>
 *  Yan Yuwen <ybbaigo@gmail.com>
 */
#ifndef __MAX_ENT_MAP__
#define __MAX_ENT_MAP__
#undef  __DEPRECATED

#include <ext/hash_map>
#include <vector>

namespace maxent
{

class MaxEntMap
{
public:

    MaxEntMap();
    ~MaxEntMap();

public:
    size_t insertString(const std::string & str);
    size_t str2num(const std::string & str) const;
    std::string & num2str(const size_t Id, std::string & str) const;

private:
    struct string_hash {
        size_t operator()(const std::string & str) const{
            return __gnu_cxx::__stl_hash_string(str.c_str());
        }
    };

private:
    __gnu_cxx::hash_map<std::string, size_t, string_hash> mStr2Num;
    std::vector<std::string> mNum2Str;
    size_t mNumber; // auto increment
};

}

#endif
