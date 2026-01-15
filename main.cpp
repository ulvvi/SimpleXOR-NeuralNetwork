#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

/*
* These two functions are essential for providing the neural network the ability to comprehend further than simplistic linear functions
* :)
*/
// Activation function
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
// Backpropag
double sigmoid_derivative(double x) { return x * (1.0 - x); }


//main layer struct
struct layer {
	//size of the layer that will input towards this one
	int inputSize;
	//number of neurons in this layer
	int neuronCount;

	//the weights and neuron mutiplication can be stored in a matrix
	//weights[neuron][input}
	vector<vector<double>> weights;

	//biases of the neurons in the layer
	vector<double> biases;
	//the outputs of the neurons in this layer
	//used for foward prop
	vector<double> outputs;
	//same but used for back prop
	vector<double> deltas;



	//initialize the layer with values
	void initLayer(int numInputs, int numNeurons) {
		weights.resize(numNeurons, vector<double>(numInputs));
		biases.resize(numNeurons);
		outputs.resize(numNeurons);
		deltas.resize(numNeurons);

		/*
		* testing
		* initializes with random values;
		*/
		for (int i = 0; i < numNeurons; i++) {
			biases[i] = ((double)rand() / RAND_MAX) * 2 - 1; //this is so that range is from -1 to 1
			for (int j = 0; j < numInputs; j++) {
				weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; //same thing
			}
		}
	}
};

//this function is responsible for the foward propagation of a layer :)
void fowardProp(layer& layer, const vector<double>& inputs) {
	for (int i = 0; i < layer.weights.size(); i++) {
		double activation = layer.biases[i];
		for (int j = 0; j < inputs.size(); j++) {
			activation += inputs[j] * layer.weights[i][j];
		}

		layer.outputs[i] = sigmoid(activation); // outputs are the result of the calculations above injected into sigmoid func
	}
}


int main() {
	srand(time(0)); // random seed is the PC's time

	//XOR data
	//should recieve these values and return the out values
	vector<vector<double>> In_data = { {0,0}, {0,1}, {1,0}, {1,1} };
	vector<double> Out_data = { 0, 1, 1, 0 };


	//input layer may be omitted because its inputs are already accounted for on the hiddenlayer
	layer hiddenLayer;
	hiddenLayer.initLayer(2, 2); //2 inputs and 2 outputs


	layer outputLayer;
	outputLayer.initLayer(2, 1); //2 inputs to final output

	double learningRate = 0.2;

	//epoch is how much we want to repeat this single "experiment"
	//ex epoch will make this repeat 20000 times
	for (int epoch = 0; epoch < 20000; epoch++) {
		double total_error = 0; // track the total error
		//learn from the 4 possible results
		for (int i = 0; i < 4; i++) {
			//foward prop
			fowardProp(hiddenLayer, In_data[i]); //will iterate through the 4 possiple input combinations
			fowardProp(outputLayer, hiddenLayer.outputs);

			//backward prop
			// Output layer's delta, basically the error margin that it got wrong
			double error = Out_data[i] - outputLayer.outputs[0];
			total_error += error * error;
			outputLayer.deltas[0] = error * sigmoid_derivative(outputLayer.outputs[0]);

			//same thing but for the hidden layer
			for (int j = 0; j < hiddenLayer.outputs.size(); j++) {
				double hiddenError = outputLayer.deltas[0] * outputLayer.weights[0][j];
				hiddenLayer.deltas[j] = hiddenError * sigmoid_derivative(hiddenLayer.outputs[j]); //iterates back through the chain rule (CALCULUS 1!!!)
			}

			//update weights for output based on who is to blame
			for (int j = 0; j < hiddenLayer.outputs.size(); j++) {
				outputLayer.weights[0][j] += learningRate * outputLayer.deltas[0] * hiddenLayer.outputs[j];
			}
			outputLayer.biases[0] += learningRate * outputLayer.deltas[0];

			//same thing but for the hidden layer
			for (int j = 0; j < hiddenLayer.outputs.size(); j++) {
				for (int k = 0; k < 2; k++) {
					hiddenLayer.weights[j][k] += learningRate * hiddenLayer.deltas[j] * In_data[i][k];
				}
				hiddenLayer.biases[j] += learningRate * hiddenLayer.deltas[j]; //update biases based on learning rate and who got it wrong
			}

			//this is only for fun and allows me to see the progress of the neural net
			if (epoch % 1000 == 0) {
				cout << "Epoch: " << epoch << " || Total Error: " << total_error << endl;

				//tries a pattern to see
				fowardProp(hiddenLayer, { 1.0, 1.0 });
				fowardProp(outputLayer, hiddenLayer.outputs);
				cout << "   Current guess for (1,1): " << outputLayer.outputs[0] << endl;
				cout << "------------------------------------------" << endl;
			}
		}
	}

	return 0;
}