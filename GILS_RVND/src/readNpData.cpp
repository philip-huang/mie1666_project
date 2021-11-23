#include "cnpy.h"
#include "readData.h"
#include <iostream>

using namespace std;

void readNpData( int argc, char** argv, int* Dimension, double ***Mdist ) {
    if (argc < 2) {
        cout << "\nNot Enough parameters\n";
        cout << " ./exec [path to npz instance] "<< endl;
        exit(1);
    }

    if (argc > 2) {
        cout << "\nToo many parameters\n";
        cout << " ./exec [path to npz instance] " << endl;
        exit(1);
    }

    char *path = argv[1];
    cnpy::npz_t npz_t = cnpy::npz_load(path);
    cnpy::NpyArray arr = npz_t["cost_matrix"];
    double* loaded_data = arr.data<double>();

    assert(arr.word_size == sizeof(double));
    assert(arr.shape.size() == 2);
    int N = arr.shape[0];
    int M = arr.shape[1];
    assert(N == M);

    *Dimension = N;
    double **dist = new double*[N+1];
    
    for ( int i = 1; i < N + 1; i++ ) {
        dist [i] = new double [N+1];
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            dist[i+1][j+1] = loaded_data[i * N + j];
            std::cout << dist[i+1][j+1] << " ";
        }
        std::cout << std::endl;
    }

    *Mdist = dist;


}
