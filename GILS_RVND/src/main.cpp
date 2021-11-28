#include "readData.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>
#include <cfloat>
#include <algorithm>

using namespace std;

double ** distanceMatrix; 
int dimension; 

struct insertionInfo{
  int insertedNode;
  int deletedEdge; 
  double cost;
};

struct neighborInfo{
  int iBest;
  int jBest;
  double bestCost;
};

struct subsequenceInfo{
  double totalTime;
  double acumulateCost;
  double vertices;
};

bool compares(insertionInfo a, insertionInfo b){
  return a.cost < b.cost;
}

neighborInfo swap(vector <int> &solution, vector <vector <subsequenceInfo>> &subsequenceMatrix){
  double partialCost, cost, partialTime;
  int size = solution.size();

  neighborInfo bestNeighbor;
  bestNeighbor.bestCost = DBL_MAX;

  for(int i = 1; i < size-2; i++){
    for(int j = i+1; j < size-1; j++){
      if(j == i + 1){
        partialCost = subsequenceMatrix[0][i-1].acumulateCost + subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[j]];
        partialTime = subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[j]];

        partialCost = partialCost + partialTime + distanceMatrix[solution[j]][solution[i]];
        partialTime = partialTime + distanceMatrix[solution[j]][solution[i]];

        cost = partialCost + ((dimension-j) * (partialTime + distanceMatrix[solution[i]][solution[j+1]])) + subsequenceMatrix[j+1][dimension].acumulateCost;
      }else{
        partialCost = subsequenceMatrix[0][i-1].acumulateCost + subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[j]];
        partialTime = subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[j]];

        partialCost = partialCost + partialTime + distanceMatrix[solution[j]][solution[i+1]];
        partialTime = partialTime + distanceMatrix[solution[j]][solution[i+1]];

        partialCost = partialCost + ((j-i-2) * (partialTime)) + subsequenceMatrix[i+1][j-1].acumulateCost;
        partialTime = partialTime + subsequenceMatrix[i+1][j-1].totalTime;

        partialCost = partialCost + partialTime + distanceMatrix[solution[j-1]][solution[i]];
        partialTime = partialTime + distanceMatrix[solution[j-1]][solution[i]];

        cost = partialCost + ((dimension-j) * (partialTime + distanceMatrix[solution[i]][solution[j+1]])) + subsequenceMatrix[j+1][dimension].acumulateCost;
      }

      if(cost < bestNeighbor.bestCost){    
        bestNeighbor.iBest = i;
        bestNeighbor.jBest = j;
        bestNeighbor.bestCost = cost;
	    } 
    }
  }

  return bestNeighbor;
}

neighborInfo twoOpt(vector <int> &solution, vector <vector <subsequenceInfo>> &subsequenceMatrix){
  double partialCost, cost, partialTime;
  int size = solution.size();

  neighborInfo bestNeighbor;
  bestNeighbor.bestCost = DBL_MAX;

  for(int i = 1; i < size; i++){
    for(int j = i+1; j < size-1; j++){
      partialCost = subsequenceMatrix[0][i-1].acumulateCost + ((j-i+1) * (subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[j]])) + subsequenceMatrix[j][i].acumulateCost;
      partialTime = subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[j]] + subsequenceMatrix[j][i].totalTime;

      cost = partialCost + ((dimension-j) * (partialTime + distanceMatrix[solution[i]][solution[j+1]]) + subsequenceMatrix[j+1][dimension].acumulateCost);

      if(cost < bestNeighbor.bestCost){    
        bestNeighbor.iBest = i;
        bestNeighbor.jBest = j;
        bestNeighbor.bestCost = cost;
	    } 
    }
  }

  return bestNeighbor;
}

neighborInfo reinsertion(vector <int> &solution, vector <vector <subsequenceInfo>> &subsequenceMatrix){
  double partialCost, cost, partialTime;
  int size = solution.size();

  neighborInfo bestNeighbor;
  bestNeighbor.bestCost = DBL_MAX;

  for(int i = 1; i < size-2; i++){
    for(int j = i + 1; j < size-1; j++){
      partialCost = subsequenceMatrix[0][i-1].acumulateCost + subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[i+1]];
      partialTime = subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[i+1]];

      partialCost = partialCost + ((j-i-1) * (partialTime)) + subsequenceMatrix[i+1][j].acumulateCost;
      partialTime = partialTime + subsequenceMatrix[i+1][j].totalTime;

      partialCost = partialCost + partialTime + distanceMatrix[solution[j]][solution[i]];
      partialTime = partialTime + distanceMatrix[solution[j]][solution[i]];

      cost = partialCost + ((dimension-j) * (partialTime + distanceMatrix[solution[i]][solution[j+1]])) + subsequenceMatrix[j+1][dimension].acumulateCost;

      if(cost < bestNeighbor.bestCost){    
        bestNeighbor.iBest = i;
        bestNeighbor.jBest = j;
        bestNeighbor.bestCost = cost;
	    } 
    }
  }

  for(int j = 1; j < size-2; j++){
    for(int i = j + 1; i < size-1; i++){
      partialCost = subsequenceMatrix[0][j-1].acumulateCost + subsequenceMatrix[0][j-1].totalTime + distanceMatrix[solution[j-1]][solution[i]];
      partialTime = subsequenceMatrix[0][j-1].totalTime + distanceMatrix[solution[j-1]][solution[i]];

      partialCost = partialCost + ((i-j) * (partialTime + distanceMatrix[solution[i]][solution[j]])) + subsequenceMatrix[j][i-1].acumulateCost;
      partialTime = partialTime + distanceMatrix[solution[i]][solution[j]] + subsequenceMatrix[j][i-1].totalTime;

      cost = partialCost + ((dimension-i) * (partialTime + distanceMatrix[solution[i-1]][solution[i+1]])) + subsequenceMatrix[i+1][dimension].acumulateCost;

      if(cost < bestNeighbor.bestCost){    
        bestNeighbor.iBest = i;
        bestNeighbor.jBest = j;
        bestNeighbor.bestCost = cost;
	    } 
    }
  }

  return bestNeighbor;
}

neighborInfo oropt2(vector <int> &solution, vector <vector <subsequenceInfo>> &subsequenceMatrix){
  double partialCost, cost, partialTime;
  int size = solution.size();

  neighborInfo bestNeighbor;
  bestNeighbor.bestCost = DBL_MAX;

  for(int i = 1; i < size-2; i++){
    for(int j = 1; j <= size-3; j++){
      if(i != j){
        if(i < j){
          partialCost = subsequenceMatrix[0][i-1].acumulateCost + subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[i+2]];
          partialTime = subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[i+2]];

          partialCost = partialCost + ((j-i-1) * (partialTime)) + subsequenceMatrix[i+2][j+1].acumulateCost;
          partialTime = partialTime + subsequenceMatrix[i+2][j+1].totalTime;

          partialCost = partialCost + partialTime + distanceMatrix[solution[j+1]][solution[i]];
          partialTime = partialTime + distanceMatrix[solution[j+1]][solution[i]];

          partialCost = partialCost + partialTime + distanceMatrix[solution[i]][solution[i+1]];
          partialTime = partialTime + distanceMatrix[solution[i]][solution[i+1]];

          cost = partialCost + ((dimension-j-1) * (partialTime + distanceMatrix[solution[i+1]][solution[j+2]])) + subsequenceMatrix[j+2][dimension].acumulateCost;
        }else{
          partialCost = subsequenceMatrix[0][j-1].acumulateCost + subsequenceMatrix[0][j-1].totalTime + distanceMatrix[solution[j-1]][solution[i]];
          partialTime = subsequenceMatrix[0][j-1].totalTime + distanceMatrix[solution[j-1]][solution[i]];

          partialCost = partialCost + partialTime + distanceMatrix[solution[i]][solution[i+1]];
          partialTime = partialTime + distanceMatrix[solution[i]][solution[i+1]];

          partialCost = partialCost + partialTime + distanceMatrix[solution[i+1]][solution[j]];
          partialTime = partialTime + distanceMatrix[solution[i+1]][solution[j]];

          partialCost = partialCost + ((i-j-1) * (partialTime)) + subsequenceMatrix[j][i-1].acumulateCost;
          partialTime = partialTime + subsequenceMatrix[j][i-1].totalTime;

          cost = partialCost + ((dimension-i-1) * (partialTime + distanceMatrix[solution[i-1]][solution[i+2]])) + subsequenceMatrix[i+2][dimension].acumulateCost;
        }

        if(cost < bestNeighbor.bestCost){    
          bestNeighbor.iBest = i;
          bestNeighbor.jBest = j;
          bestNeighbor.bestCost = cost;
	      } 
      }
    }
  }

  return bestNeighbor;
}

neighborInfo oropt3(vector <int> &solution, vector <vector <subsequenceInfo>> &subsequenceMatrix){
  double partialCost, cost, partialTime;
  int tam = solution.size();
  neighborInfo bestNeighbor;
  bestNeighbor.bestCost = DBL_MAX;

  for(int i = 1; i < tam-3; i++){
    for(int j = 1; j <= tam-4; j++){
      if(i != j){
        if(i < j){
          partialCost = subsequenceMatrix[0][i-1].acumulateCost + subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[i+3]];
          partialTime = subsequenceMatrix[0][i-1].totalTime + distanceMatrix[solution[i-1]][solution[i+3]];

          partialCost = partialCost + ((j-i-1) * (partialTime)) + subsequenceMatrix[i+3][j+2].acumulateCost;
          partialTime = partialTime + subsequenceMatrix[i+3][j+2].totalTime;

          partialCost = partialCost + partialTime + distanceMatrix[solution[j+2]][solution[i]];
          partialTime = partialTime + distanceMatrix[solution[j+2]][solution[i]];

          partialCost = partialCost + ((2) * (partialTime)) + subsequenceMatrix[i][i+2].acumulateCost;
          partialTime = partialTime + subsequenceMatrix[i][i+2].totalTime;

          cost = partialCost + ((dimension-j-2) * (partialTime + distanceMatrix[solution[i+2]][solution[j+3]])) + subsequenceMatrix[j+3][dimension].acumulateCost;
        }else{
          partialCost = subsequenceMatrix[0][j-1].acumulateCost + subsequenceMatrix[0][j-1].totalTime + distanceMatrix[solution[j-1]][solution[i]];
          partialTime = subsequenceMatrix[0][j-1].totalTime + distanceMatrix[solution[j-1]][solution[i]];

          partialCost = partialCost + ((2) * (partialTime)) + subsequenceMatrix[i][i+2].acumulateCost;
          partialTime = partialTime + subsequenceMatrix[i][i+2].totalTime;

          partialCost = partialCost + partialTime + distanceMatrix[solution[i+2]][solution[j]];
          partialTime = partialTime + distanceMatrix[solution[i+2]][solution[j]];

          partialCost = partialCost + ((i-j-1) * (partialTime)) + subsequenceMatrix[j][i-1].acumulateCost;
          partialTime = partialTime + subsequenceMatrix[j][i-1].totalTime;

          cost = partialCost + ((dimension-i-2) * (partialTime + distanceMatrix[solution[i-1]][solution[i+3]])) + subsequenceMatrix[i+3][dimension].acumulateCost;
        }

        if(cost < bestNeighbor.bestCost){    
          bestNeighbor.iBest = i;
          bestNeighbor.jBest = j;
          bestNeighbor.bestCost = cost;
	      }
      }
    }
  }

  return bestNeighbor;
}

// Function to update subsequence matrix with a new solution
void updatesMatrix(vector <vector <subsequenceInfo>> &subsequenceMatrix, vector <int> &solution){
  int tam = solution.size();

  // Updates total time
  for(int i = 0; i < tam; i++){
    for(int j = i; j < tam; j++){
      if(i == j){
        subsequenceMatrix[i][j].totalTime = 0;
      }else {
        subsequenceMatrix[i][j].totalTime = subsequenceMatrix[i][j-1].totalTime + distanceMatrix[solution[j-1]][solution[j]];
        subsequenceMatrix[j][i].totalTime = subsequenceMatrix[j-1][i].totalTime + distanceMatrix[solution[j]][solution[j-1]];  
      }
    }
  }

  // Updates acumulate cost
  for(int i = 0; i < tam; i++){
    for(int j = i; j < tam; j++){ 
      if(i == j){
        subsequenceMatrix[i][j].acumulateCost = 0;
      }else {
        subsequenceMatrix[i][j].acumulateCost = subsequenceMatrix[i][j-1].acumulateCost + subsequenceMatrix[i][j].totalTime;
      }      
    }
  }

  for(int i = tam-1; i >= 0; i--){
    for(int j = i; j >= 0; j--){
      if(i == j){
        subsequenceMatrix[i][j].acumulateCost = 0;
      }else{
        subsequenceMatrix[i][j].acumulateCost = subsequenceMatrix[i][j+1].acumulateCost + subsequenceMatrix[i][j].totalTime;
      }      
    }
  }
}

void RVND(vector <int> &solution, vector <vector <subsequenceInfo>> &subsequenceMatrix){
  vector <int> neighborhoods = {0, 1, 2, 3, 4};
  neighborInfo neighbor;

  while(!neighborhoods.empty()){
    int choosen = rand() % neighborhoods.size();

    if(neighborhoods[choosen] == 0){
      neighbor = swap(solution, subsequenceMatrix);

      if(neighbor.bestCost < subsequenceMatrix[0][dimension].acumulateCost){

        int aux = solution[neighbor.iBest];
	      solution[neighbor.iBest] = solution[neighbor.jBest];
	      solution[neighbor.jBest] = aux;

        // Updates subsequence matrix with new solution
        updatesMatrix(subsequenceMatrix, solution);
        neighborhoods = {0, 1, 2, 3, 4};

      } else {
        neighborhoods.erase(neighborhoods.begin() + choosen);

      }

    }else if(neighborhoods[choosen] == 1){
      neighbor = reinsertion(solution, subsequenceMatrix);

      if(neighbor.bestCost < subsequenceMatrix[0][dimension].acumulateCost){
        vector <int> solutionInicial = solution;

        solution.erase(solution.begin()+neighbor.iBest);
        solution.insert(solution.begin()+neighbor.jBest, solutionInicial[neighbor.iBest]);

        // Updates subsequence matrix with new solution
        updatesMatrix(subsequenceMatrix, solution);
        neighborhoods = {0, 1, 2, 3, 4};

      } else {
        neighborhoods.erase(neighborhoods.begin() + choosen);

      }

    }else if(neighborhoods[choosen] == 2){
      neighbor = twoOpt(solution, subsequenceMatrix);

      if(neighbor.bestCost < subsequenceMatrix[0][dimension].acumulateCost){

        int aux, k = neighbor.jBest - neighbor.iBest;

        if(k % 2 != 0){
          k = k + 1;
        }

        for(int q = 0; q < k/2; q++){
          aux = solution[neighbor.iBest+q];
          solution[neighbor.iBest+q] = solution[neighbor.jBest-q];
          solution[neighbor.jBest-q] = aux;
        }

        // Updates subsequence matrix with new solution
        updatesMatrix(subsequenceMatrix, solution);
        neighborhoods = {0, 1, 2, 3, 4};

      } else {
        neighborhoods.erase(neighborhoods.begin() + choosen);

      }

    }else if(neighborhoods[choosen] == 3){

      neighbor = oropt2(solution, subsequenceMatrix);

      if(neighbor.bestCost < subsequenceMatrix[0][dimension].acumulateCost){

        if(neighbor.iBest < neighbor.jBest){
          solution.insert(solution.begin() + neighbor.jBest + 2, solution[neighbor.iBest]); 
          solution.insert(solution.begin() + neighbor.jBest + 3, solution[neighbor.iBest+1]); 
          solution.erase(solution.begin() + neighbor.iBest);
          solution.erase(solution.begin() + neighbor.iBest);
        } else {
          solution.insert(solution.begin() + neighbor.jBest, solution[neighbor.iBest]); 
          solution.insert(solution.begin() + neighbor.jBest + 1, solution[neighbor.iBest + 2]); 
          solution.erase(solution.begin() + neighbor.iBest + 2);
          solution.erase(solution.begin() + neighbor.iBest + 2);
        }

        // Updates subsequence matrix with new solution
        updatesMatrix(subsequenceMatrix, solution);
        neighborhoods = {0, 1, 2, 3, 4};

      } else {
        neighborhoods.erase(neighborhoods.begin() + choosen);
      }   

    }else if(neighborhoods[choosen] == 4){
      neighbor = oropt3(solution, subsequenceMatrix);

      if(neighbor.bestCost < subsequenceMatrix[0][dimension].acumulateCost){

        if(neighbor.iBest < neighbor.jBest){
          solution.insert(solution.begin() + neighbor.jBest + 3, solution[neighbor.iBest]);
          solution.insert(solution.begin() + neighbor.jBest + 4, solution[neighbor.iBest+1]); 
          solution.insert(solution.begin() + neighbor.jBest + 5, solution[neighbor.iBest+2]);
          solution.erase(solution.begin() + neighbor.iBest);
          solution.erase(solution.begin() + neighbor.iBest);
          solution.erase(solution.begin() + neighbor.iBest);
        } else {
          solution.insert(solution.begin() + neighbor.jBest, solution[neighbor.iBest]);
          solution.insert(solution.begin() + neighbor.jBest + 1, solution[neighbor.iBest + 2]); 
          solution.insert(solution.begin() + neighbor.jBest + 2, solution[neighbor.iBest + 4]); 
          solution.erase(solution.begin() + neighbor.iBest + 3);
          solution.erase(solution.begin() + neighbor.iBest + 3);
          solution.erase(solution.begin() + neighbor.iBest + 3);
        }

        // Updates subsequence matrix with new solution
        updatesMatrix(subsequenceMatrix, solution);
        neighborhoods = {0, 1, 2, 3, 4};

      } else {
        neighborhoods.erase(neighborhoods.begin() + choosen);

      }      
    }
  }
}

vector <int> pertub(vector <int> &solution){
  vector <int> newSolution;
	vector <int> firstSubsequence;
	vector <int> secondSubsequence;
	int i, j, sizeFirst = 0, sizeSecond = 0, size = solution.size();
	
	// Generates subsequence sizes
	if(size < 20){
		sizeFirst = 2;
		sizeSecond = 2;
	} else {
		while(sizeFirst < 2 || sizeFirst > (size/10)) sizeFirst = rand() % size;
		while(sizeSecond < 2 || sizeSecond > (size/10)) sizeSecond = rand() % size; 
	}
	
	// Generates initial position of first subsequence
	i = rand() % (size - sizeFirst);
  while(i == 0) i = rand() % (size - sizeFirst);

	// Generates initial position of second subsequence
	j = rand() % (size - sizeSecond);
	while((j > (i - sizeSecond) && j < (i + sizeFirst)) || j == 0) j = rand() % (size - sizeSecond);

	// Creates vector with first subsequence
	int iter = 0;
	for(int k = 0; k < size; k++){
		if(k >= i){
			firstSubsequence.push_back(solution[k]);
      iter++;
			if(iter == sizeFirst) break;
		}
	}
	
	// Creates vector with second subsequence
	iter = 0;
	for(int k = 0; k < size; k++){
		if(k >= j){
			secondSubsequence.push_back(solution[k]);
      iter++;
			if(iter == sizeSecond) break;
		}
	}

	if(j < i){	
    newSolution = solution;

    // Deletes first subsequence
    int deleted = 0;
    for(int k = 0; k < size; k++){
      if(k >= i){
        newSolution.erase(newSolution.begin() + k - deleted);
        deleted++;
        if(deleted == sizeFirst) break;
      }
    }

    // Deletes second subsequence
    deleted = 0;
    for(int k = 0; k < size; k++){
      if(k >= j){
        newSolution.erase(newSolution.begin() + k - deleted);
        deleted++;
        if(deleted == sizeSecond) break;
      }
    }

    // Inserts second subsequence
    int inserted = 0;
    for(int k = 0; k < secondSubsequence.size(); k++){
      newSolution.insert(newSolution.begin() + i + inserted - sizeSecond, secondSubsequence[k]);
      inserted++;
    }
        
    // Inserts first subsequence
    inserted = 0;
    for(int k = 0; k < firstSubsequence.size(); k++){
      newSolution.insert(newSolution.begin() + j + inserted, firstSubsequence[k]);
      inserted++;
    }
      
	} else {
    newSolution = solution;

    // Deletes second subsequence
    int deleted = 0;
    for(int k = 0; k < size; k++){
      if(k >= j){
        newSolution.erase(newSolution.begin() + k - deleted);
        deleted++;
        if(deleted == sizeSecond) break;
      }
    }

    // Deletes first subsequence
    deleted = 0;
    for(int k = 0; k < size; k++){
      if(k >= i){
        newSolution.erase(newSolution.begin() + k - deleted);
        deleted++;
        if(deleted == sizeFirst) break; 
      }
	  }

    // Inserts first subsequence
    int inserted = 0;
    for(int k = 0; k < firstSubsequence.size(); k++){
      newSolution.insert(newSolution.begin() + j + inserted - sizeFirst, firstSubsequence[k]);
      inserted++;
    }
        
    // Inserts second subsequence
    inserted = 0;
    for(int k = 0; k < secondSubsequence.size(); k++){
      newSolution.insert(newSolution.begin() + i + inserted, secondSubsequence[k]);
      inserted++;
    }
  }
		
	return newSolution;	
}

vector <int> construction(vector <int> candidatesList, double alpha){
  vector <int> initialSolution;

  // Insert depot
  initialSolution.push_back(candidatesList[0]);
  initialSolution.push_back(candidatesList[0]);

  // Deletes it from candidates list
  candidatesList.erase(candidatesList.begin());

  // Chooses 3 random vertices
  for(int i = 0; i < 3; i++){
    int j = rand() % candidatesList.size();
    initialSolution.insert(initialSolution.begin()+1, candidatesList[j]); 
    candidatesList.erase(candidatesList.begin()+j);
  }

  // Calculates insertion cost of vertices on solution
  while(!candidatesList.empty()){
    vector <insertionInfo> insertionCost((initialSolution.size()-1) * candidatesList.size());

    for(int i = 0, j = 1, k = 0; i < initialSolution.size()-1; i++, j++){
      for(auto l : candidatesList){
        insertionCost[k].cost = distanceMatrix[initialSolution[i]][l] + distanceMatrix[l][initialSolution[j]] - distanceMatrix[initialSolution[i]][initialSolution[j]];
        insertionCost[k].insertedNode = l;
        insertionCost[k].deletedEdge = i;
        k++;
      }
    }

    // Orders insertion costs
    sort(insertionCost.begin(), insertionCost.end(), compares);

    int elements = alpha * insertionCost.size();
    int i = rand() % elements;

    // Inserts choosen node
    initialSolution.insert(initialSolution.begin() + (insertionCost[i].deletedEdge + 1), insertionCost[i].insertedNode);

    // Deletes inserted node from candidates list
    for(int j = 0; j < candidatesList.size(); j++){
      if(candidatesList[j] == insertionCost[i].insertedNode)
        candidatesList.erase(candidatesList.begin()+j);
    }
  }

  return initialSolution;
}

double search(int iIls, int dimension, ofstream &fout){
  double bestCurrentCost = DBL_MAX, currentCost, finalCost = DBL_MAX;
  vector <int> vertices, bestCurrentSolution, currentSolution, finalSolution;
  vector <vector <subsequenceInfo>> subsequenceMatrix(dimension+1, vector <subsequenceInfo> (dimension+1));

  // Creates vector with vertices
  for(int i = 0; i < dimension; i++){
    vertices.push_back(i+1);
  }

  for(int iterMax = 0; iterMax < 10; iterMax++){
    double alpha = (rand() % 90) / 100.0 + 0.1;

    currentSolution = construction(vertices, alpha); // Generates initial solution
    updatesMatrix(subsequenceMatrix, currentSolution);

    bestCurrentSolution = currentSolution;

    int iterIls = 0;
    while(iterIls < iIls){
      RVND(currentSolution, subsequenceMatrix);

      currentCost = subsequenceMatrix[0][dimension].acumulateCost;

      if(currentCost < bestCurrentCost){
        bestCurrentSolution = currentSolution;
        bestCurrentCost = currentCost;
        iterIls = 0;
      }

      currentSolution = pertub(bestCurrentSolution);
      updatesMatrix(subsequenceMatrix, currentSolution);

      iterIls++;
    }

    if(bestCurrentCost < finalCost){
      finalSolution = bestCurrentSolution;
      finalCost = bestCurrentCost;
    }
  }

  cout << endl << "Solution: ";
  for(int i = 1; i < finalSolution.size() - 1; i++){
    cout << finalSolution[i] - 1 << " ";
    fout << finalSolution[i] - 1 << "\n";
  }

  return finalCost;  
}

int main(int argc, char** argv) {

  clock_t start = clock(); // Starts time counting
    
  readNpData(argc, argv, &dimension, &distanceMatrix);
  std::cout << "read successful" << std::endl;
  srand(time(NULL));

  int iIls;
    
  if(dimension >= 100){
    iIls = 100;
  } else {
    iIls = dimension;
  }

  ofstream fout;
  string filename (argv[1]);;
  filename.append(".gilsrvndSol.txt");
  fout.open(filename);

  double cost = search(iIls, dimension, fout);

  // Ends time counting
  clock_t end = clock();
  double time = ((double) (end - start)) / CLOCKS_PER_SEC;

  cout << endl << "Cost: " << cost << endl;
  cout << "Time: " << time << endl << endl;
  fout << cost << endl;
  fout << time << endl;
    
  return 0;

}