#ifndef READDATA_H_INCLUDED
#define READDATA_H_INCLUDED
#include <string>
#include <vector>
extern void readData( int , char** , int* , double *** );
extern void readNpData(int, char**, int*, double***);

void readRLSol(std::vector<int> &path, double & runtime, int n, std::string file);
#endif // READDATA_H_INCLUDED
