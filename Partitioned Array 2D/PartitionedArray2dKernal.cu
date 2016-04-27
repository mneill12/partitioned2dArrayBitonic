#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cuda_runtime_api.h>


#include "writeToCSVFileHeader.h"
#include "userInputHeader.h"

void printArray(int *elements);

int stepOneThreadCount;
int blockCount;
int allOtherStepsThreadCount;
int elementCount;

int executionCount;


bool isSorted(int *elements){

	bool sorted = true;
	for (int i = 0; i < (elementCount - 1); ++i){
		if (elements[i] > elements[i + 1]){
			sorted = false;
		}
	}
	return sorted;
}

double getElapsedTime(clock_t start, clock_t stop)
{
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);

	return elapsed;
}


const int randMax = 10000;

void createUnsortedArray(int* elements){

	for (int i = 0; i < elementCount; ++i){
		elements[i] = rand() % randMax - rand() % 5;
	}

}

struct arrays
{
	int* evenArray;
	int* oddArray;
};

arrays splitArray(int* origionalElements, arrays evenOddArrays){

	int *evenCountPtr = evenOddArrays.evenArray;
	int *oddCountPtr = evenOddArrays.oddArray;
	int count = 0;
	for (int i = 0; i < elementCount; i++){
	
		if (i % 2 == 0){

			memcpy(evenCountPtr, origionalElements, sizeof(int));
		
			evenCountPtr++;
		}
		else{
			memcpy(oddCountPtr, origionalElements, sizeof(int));

			oddCountPtr++;
		}
		origionalElements++;
	}

	return evenOddArrays;
}

//Merger even and odd arrays into origional arrays
void mergeArrays(arrays evenOddArrays,  int* origionalElemens){

	int *evenCountPtr = evenOddArrays.evenArray;
	int *oddCountPtr = evenOddArrays.oddArray;
	int count = 0;

	for (int i = 0; i < elementCount; i++){
		
		if (i % 2 == 0){

			//Destination, Source, size
			memcpy(origionalElemens, evenCountPtr, sizeof(int));
			evenCountPtr++;
		}
		else{
			memcpy(origionalElemens, oddCountPtr, sizeof(int));
			oddCountPtr++;
		}


		origionalElemens++;
	}

}

bool checkEvenOddArrays(arrays evenOddArrays,  int* origionaArray){

	int evenCount = 0;
	int oddCount = 0;

	bool elementSplitCorrectly = true;

	for (int i = 0; i < elementCount; i++){
		
		if (i % 2 == 0){

			if (origionaArray[i] != evenOddArrays.evenArray[evenCount]){

				elementSplitCorrectly = false;
			}
			evenCount++;
		}

		else{

			if (origionaArray[i] != evenOddArrays.oddArray[oddCount]){

				elementSplitCorrectly = false;
			}
			oddCount++;
		}
	}

	return elementSplitCorrectly;
}

__global__ void bitonicSortAllOtherStepsSingleThreaded(int *deviceElements, int step, int phase, int compExchSize, int fullArraySize)
{
	unsigned int  halfstep, halfphase, secondIndex;

	int firstIndex = threadIdx.x + blockDim.x * blockIdx.x;

	for (int count = 0; count < fullArraySize / 2; count++){
		halfstep = step / 2;
		halfphase = phase / 2;
		secondIndex = firstIndex^halfstep;

		if ((secondIndex) > firstIndex) {
			if ((firstIndex&halfphase) == 0) {

				if (deviceElements[firstIndex] > deviceElements[secondIndex]) {
					int temp = deviceElements[firstIndex];
					deviceElements[firstIndex] = deviceElements[secondIndex];
					deviceElements[secondIndex] = temp;
				}
			}
			if ((firstIndex&halfphase) != 0) {

				if (deviceElements[firstIndex] < deviceElements[secondIndex]) {
					int temp = deviceElements[firstIndex];
					deviceElements[firstIndex] = deviceElements[secondIndex];
					deviceElements[secondIndex] = temp;
				}
			}
		}
		firstIndex++;
	}
}

/*	As we're complying with the origional model but with arrays reduced in size by two 
	here we'll just divide firstIndex and secondIndex by two to get our required values.
	We're also going to go though all the compaire/exchange operations that would normally be carried out in this step.
*/
__global__ void bitonicSortAllOtherSteps(int *deviceElements, int step, int phase, int compExchSize, int fullArraySize)
{
	unsigned int  halfstep, halfphase, secondIndex;

	int firstIndex = threadIdx.x + blockDim.x * blockIdx.x;

	halfstep = step/2;
	halfphase = phase/2;
	secondIndex = firstIndex^halfstep;

	if ((secondIndex) > firstIndex) {
		if ((firstIndex&halfphase) == 0) {
			if (deviceElements[firstIndex] > deviceElements[secondIndex]) {
				int temp = deviceElements[firstIndex];
				deviceElements[firstIndex] = deviceElements[secondIndex];
				deviceElements[secondIndex] = temp;
				}
		}
		if ((firstIndex&halfphase) != 0) {

			if (deviceElements[firstIndex] < deviceElements[secondIndex]) {
				int temp = deviceElements[firstIndex];
				deviceElements[firstIndex] = deviceElements[secondIndex];
				deviceElements[secondIndex] = temp;
			}
		}

	}
}

__global__ void bitonicSortFirstStep(int *deviceElements, int step, int phase)
{
	unsigned int firstIndex, secondIndex; 
	firstIndex = threadIdx.x + blockDim.x * blockIdx.x;

	secondIndex = firstIndex^step;

	if ((secondIndex)>firstIndex) {
		if ((firstIndex& phase) == 0) {

			if (deviceElements[firstIndex]>deviceElements[secondIndex]) {
				int temp = deviceElements[firstIndex];
				deviceElements[firstIndex] = deviceElements[secondIndex];
				deviceElements[secondIndex] = temp;
			}
		}
		if ((firstIndex&phase) != 0) {

			if (deviceElements[firstIndex]<deviceElements[secondIndex]) {

				int temp = deviceElements[firstIndex];
				deviceElements[firstIndex] = deviceElements[secondIndex];
				deviceElements[secondIndex] = temp;
			}
		}
	}
}


//Launcher function for our instep inplmentation of bitonic sort
void bitonic_sort(int *values)
{
	int *deviceElements;
	int *deviceEvenArray;
	int *deviceOddArray;
	size_t size = elementCount * sizeof(int);

	size_t evenOddSize = elementCount/2 * sizeof(int);

	arrays evenOddArrays;


	//Allocate half of element size to each of the odd and even arrays
	evenOddArrays.evenArray = (int*)malloc((elementCount)* sizeof(int));
	evenOddArrays.oddArray = (int*)malloc((elementCount)* sizeof(int));

	cudaMalloc((void**)&deviceElements, size);
	cudaMalloc((void**)&deviceEvenArray, evenOddSize);
	cudaMalloc((void**)&deviceOddArray, evenOddSize);

	dim3 blocks(blockCount, 1);    
	dim3 stepOneThreads(stepOneThreadCount, 1);

	dim3 allOtherStepThreads(allOtherStepsThreadCount, 1);

	int compExchCount = (elementCount / 4);

	int step, phase;

	for (phase = 2; phase <= elementCount; phase <<= 1) {
		evenOddArrays = splitArray(values, evenOddArrays);

		cudaMemcpy(deviceEvenArray, evenOddArrays.evenArray, evenOddSize, cudaMemcpyHostToDevice);
		cudaMemcpy(deviceOddArray, evenOddArrays.oddArray, evenOddSize, cudaMemcpyHostToDevice);

		for (step = phase >> 1; step > 0 ; step = step >> 1) {

			if (step != 1){
			
				//Even and odd arrays to kernals 			
				bitonicSortAllOtherSteps << <blocks, allOtherStepThreads >> >(deviceEvenArray, step, phase, compExchCount, elementCount);
				bitonicSortAllOtherSteps << <blocks, allOtherStepThreads >> >(deviceOddArray, step, phase, compExchCount, elementCount);
				
			}

			//The last step, so copy back the sorted even odd arrays, merge them into the origional element array copy that to memory then sort it 
			else{
			
				cudaMemcpy(evenOddArrays.evenArray, deviceEvenArray, evenOddSize, cudaMemcpyDeviceToHost);
				cudaMemcpy(evenOddArrays.oddArray, deviceOddArray, evenOddSize, cudaMemcpyDeviceToHost);

				mergeArrays(evenOddArrays, values);
	
				cudaMemcpy(deviceElements, values, size, cudaMemcpyHostToDevice);
				bitonicSortFirstStep << <blocks, stepOneThreads >> >(deviceElements, step, phase);
	
				cudaMemcpy(values, deviceElements, size, cudaMemcpyDeviceToHost);

			}
		}
	}

	cudaFree(deviceElements);
	cudaFree(deviceEvenArray);
	cudaFree(deviceOddArray);

}

void preExecution(){

	int values[7];
	values[0] = 10;
	values[1] = 13;
	values[2] = 9;
	values[3] = 18;
	values[4] = 26;
	values[4] = 100;
	values[6] = 3;

	bitonic_sort(values);
}

int main(void)
{

	executionCount = getMaxProcessCount();
	int fixedExecutionCount = executionCount;

	preExecution();

	bool runSort = true;

	//Pointers to store our results that we're writing to CSV files, allocate space entered buy the user
	int* threadCounts = (int*)malloc(executionCount*sizeof(int));
	int* allBlocks = (int*)malloc(executionCount*sizeof(int));;
	double* timeResults = (double*)malloc(executionCount*sizeof(double));;
	char* arrayStates = (char*)malloc(executionCount*sizeof(char));

	double time;
	clock_t start, stop;
	//Counter so we can assine values to the array in the execution loop

	while (runSort && executionCount != 0){

		runSort = runSortAgain();

		//Get thread, blocks and  element count

		//Get total elements and suggested block thread configurations
		blockAndThreadCounts inputCountandSuggestedThreadBlockCount;
		inputCountandSuggestedThreadBlockCount = getElementCounts();
		elementCount = inputCountandSuggestedThreadBlockCount.elementCount;

		//wirte possible thread and block configurations to text file
		printf("Writing suggested block thread configuration...");
		writeSuggestedBlockThreadConfigToCsv(inputCountandSuggestedThreadBlockCount.threadCounts,
			inputCountandSuggestedThreadBlockCount.blockCounts,
			inputCountandSuggestedThreadBlockCount.combinationsCount
			);
		printf("Done \n");

		//Get block count and thread count and thena assign half that thread count for all other steps
	    blockCount = getBlockCount();
		stepOneThreadCount = getThreadCount();
		allOtherStepsThreadCount = stepOneThreadCount / 2;


		//Malloc array, add values to it and write unsorted array to csv file
		int* values = (int*)malloc(elementCount*sizeof(int));
		createUnsortedArray(values);
		writeBlockElementCsvFile(values, "preSorted", stepOneThreadCount, blockCount);

		//Do Sort and time it
		start = clock();
		bitonic_sort(values);
		stop = clock();

		time = getElapsedTime(start, stop);

		char* arrayState;
		char arrayStateChar;

		if (isSorted(values)){

			printf("Is Sorted \n");
			arrayState = "sorted";
			arrayStateChar = 's';
		}
		else{

			printf("Not Sorted \n");
			arrayState = "unsorted";
			arrayStateChar = 'u';
		}

		writeBlockElementCsvFile(values, arrayState, stepOneThreadCount, blockCount);

		//Allocate results values to pointers 
		*threadCounts = stepOneThreadCount;
		*allBlocks = blockCount;
		*timeResults = time;
		*arrayStates = arrayStateChar;

		//Increment Result pointers
		threadCounts++;
		allBlocks++;
		timeResults++;
		arrayStates++;

		free(values);

		//Check again for user input

		executionCount--;
	}

	printf("Execution ended. Writing results to C:\BitonicSortArrayCSVFiles /n");

	writeSortResultsToCsv(timeResults, "PartitionedArray2DBitonicSort", arrayStates, threadCounts, allBlocks, fixedExecutionCount);

	getchar();

}