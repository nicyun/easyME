// a sample program to use easy-me toolkit
#include <iostream>
#include <fstream>
#include <sstream>

#include "../src/maxEntModel.h"

using namespace std;
using namespace maxent;

int main(int argc, char * argv[])
{
    if(argc < 3){
		cerr << "Usage: " << argv[0] << " train_file_name model_file_name [num_iter] [training_algorithm] [tol] [sigma2] [alpha]" << endl;
        exit(-1);
    }

    size_t iter = 100;
    string method = "LBFGS";
    double tol = 1e-3;
    double sigma2 = 100;
    double alpha = 0;

    if(argc > 3) iter = atoi(argv[3]);
    if(argc > 4) method = argv[4];
    if(argc > 5) tol = atof(argv[5]);
    if(argc > 6) sigma2 = atof(argv[6]);
    if(argc > 7) alpha = atof(argv[7]);
	cout << "iter = " << iter << endl;
	cout << "method = " << method << endl;
	cout << "tol = " << tol << endl;
	cout << "sigma2 = " << sigma2 << endl;
	cout << "alpha = " << alpha << endl;

    MaxEntModel * me = new MaxEntModel;
	int st = clock();
    me->initModel(argv[1], 0, 0);
	me->trainModel(iter, method, tol, sigma2, alpha);
   	int tt = clock();
	cout << "time consumed by training is : " << (double)(tt - st) / CLOCKS_PER_SEC << endl;
	me->saveModel(argv[2]);
    delete me;
	return 0;
}
