#include <iostream>
#include <fstream>
#include <sstream>

#include "maxEntModel.h"

using namespace std;
using namespace maxent;

MaxEntModel * me;

void testModel(const char * testFile)
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
    cout << "just predict time is : " << preTime / CLOCKS_PER_SEC << "s" << endl;
	cout << "test ok and corr is : " << (double)corr / tot << endl;
}


int main(int argc, char * argv[])
{
    const char * trainFile = "../testdata/train";
    const char * testFile = "../testdata/test";

    size_t iter = 100;
    string method = "LBFGS";
    double tol = 1e-3;
    double sigma2 = 100;
    double alpha = 0;
    bool isTrain = 1;

    if(argc > 1) iter = atoi(argv[1]);
    if(argc > 2) method = argv[2];
    if(argc > 3) tol = atof(argv[3]);
    if(argc > 4) sigma2 = atof(argv[4]);
    if(argc > 5) alpha = atof(argv[5]);
    if(argc > 6) isTrain = atoi(argv[6]);
	if(isTrain){
		cout << "iter = " << iter << endl;
		cout << "method = " << method << endl;
		cout << "tol = " << tol << endl;
		cout << "sigma2 = " << sigma2 << endl;
		cout << "alpha = " << alpha << endl;
	}

    bool select = 0;
    if(isTrain){
    	me = new MaxEntModel;
	int st = clock();
        me->initModel(trainFile, 0, select);
	me->trainModel(iter, method, tol, sigma2, alpha);
       	int tt = clock();
	cout << "train time is : " << (double)(tt - st) / CLOCKS_PER_SEC << endl;
	me->saveModel();
    	delete me;
    }
    me = new MaxEntModel;
    // get load file and map times consumed
    int stime = clock();
    me->loadModel();
    int ttime = clock();
    cout << "file load and string map consumes " << (double)(ttime - stime) / CLOCKS_PER_SEC << "s" << endl;
    testModel(testFile);
    delete me;
	return 0;
}
