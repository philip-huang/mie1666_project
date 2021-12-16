#include "readData.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
void readRLSol(std::vector<int> &path, double &runtime, int n, std::string file)
{
    ifstream in(file, std::ifstream::in);
    if (!in)
    {
        cout << "Error opening file" << endl;
        exit(1);
    }

    // insert node 1
    path.push_back(1);
    int node;
    for (int i = 0; i < n; i++) {
        in >> node;
        path.push_back(node + 1);
    }
    in >> runtime;
    in >> runtime;
    in.close();
}
