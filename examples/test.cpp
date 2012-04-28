// a sample program to use easy-me toolkit
#include <iostream>
#include <fstream>
#include <sstream>

#include "../src/maxEntModel.h"

using namespace std;
using namespace maxent;

void testModel(MaxEntModel * me, const char * testFile)
{
    string line, str;
    ifstream fin(testFile);
    if(!fin){
        cerr << "test file not found!" << endl;
        exit(-1);
    }

    string className;
    vector<string> context;
    vector<pair<string, double> > outcome;
    int tot = 0, corr = 0;
    // each line is a test event
    cout << "start testing ..." << endl;
    int stime = clock();
    double preTime = 0;
    while(getline(fin, line)){
        istringstream sin(line);
        sin >> className;
        context.clear();
        while(sin >> str){
            context.push_back(str);
        }
        double t1 = clock();
        size_t best = me->predict(context, outcome);
        //string res;
        //me->predict(res, context);
        double t2 = clock();
        preTime += t2 - t1;
        tot++;
        if(outcome[best].first == className) corr++;
        //if(res == className) corr++;
    }
    int ttime = clock();
    cout << "all test time (including read file time) is : " << (double)(ttime - stime) / CLOCKS_PER_SEC << "s" << endl;
    cout << "only predict time is : " << preTime / CLOCKS_PER_SEC << "s" << endl;
	cout << "test ok and corr is : " << (double)corr / tot << endl;
}


int main(int argc, char * argv[])
{
    if(argc < 3){
        cerr << "please specify the model file name and test file name!" << endl;
        exit(-1);
    }
    MaxEntModel * me = new MaxEntModel;
    // get load file and map times consumed
    int stime = clock();
    me->loadModel(argv[1]);
    int ttime = clock();
    cout << "file load and string map consumes: " << (double)(ttime - stime) / CLOCKS_PER_SEC << "s" << endl;
    testModel(me, argv[2]);
    ttime = clock();
    delete me;
	return 0;
}
