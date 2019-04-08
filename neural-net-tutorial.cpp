#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cassert>
#include <math.h>
#include <fstream>
#include <sstream>
using namespace std;


class TrainingData{
	public: 
		TrainingData (const string filename);
		bool isEof(void) { return m_trainingDataFile.eof(); }
		void getTopology(vector<unsigned> &topology);

		// Return 
		unsigned getNextInputs(vector<double> &inputVals);
		unsigned getTargetOutputs(vector<double> &targetOutputVals);

	private :
		ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology){
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;

	if( this-> isEof() || label.compare("topology : ") != 0){
		abort();
	}

	while(!ss.eof()){
		unsigned n; 
		ss >> n;
		topology.push_back(n);
	}

	return;
}

TrainingData::TrainingData(const string filename){
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> & inputVals) {
	inputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;

	if(label.compare("in:") == 0){
		double oneValue;
		while(ss >> oneValue){
			inputVals.push_back(oneValue);
		}
	}
	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals){
	targetOutputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;

	if(label.compare("out: ") == 0){
		double oneValue;
		while(ss >> oneValue){
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}

struct Connection{
	double weight;	
	double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

// Class Neuron =======================================
class Neuron{
	public: 
		Neuron(unsigned numOutputs, unsigned myIndex);
		void setOutputVal(double val ){ m_outputVal = val; }
		double getOutputVal(void) const{ return m_outputVal; }
		void feedForward(const Layer &prevLayer);
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const Layer &nextLayer);
		void updateInputWeights (Layer &prevLayer);

	private:
		static double learningRate; //[0.0 .. 1.0] overall net training rate
		static double momentum; //[0.0 .. n]	multiplier of last weight change 
		static double transferFunction(double x);
		static double transferFunctioDerivative(double x);
	 	static double randomWeight(void){
			return rand() / double(RAND_MAX);
		}

		double sumDOW(const Layer &nextLayer) const;
		double m_outputVal;
		vector<Connection> m_outputWeights;
		unsigned m_myIndex;
		double m_gradient;
};

double Neuron::learningRate = 0.15;
double Neuron::momentum= 0.5;

void Neuron::updateInputWeights (Layer &prevLayer){
	 //update weights in Connection container
	for(unsigned n = 0; n < prevLayer.size(); n++){
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		
		double newDeltaWeight = 
				learningRate 
				* neuron.getOutputVal()
				* m_gradient
				+ momentum
				* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const{
	double sum = 0.0;

	for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
		sum+= m_outputWeights[n].weight * nextLayer[n].m_gradient;		
	}
	return sum;
}


void Neuron::calcHiddenGradients(const Layer &nextLayer){
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctioDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal){
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctioDerivative(m_outputVal);
}

double Neuron::transferFunction(double x){
	//tanh - output : [-1.0 .. 1.0]
	return tanh(x);
}
double Neuron::transferFunctioDerivative(double x){
	//tanh derivative
	return 1.0 - x*x;
}

void Neuron::feedForward(const Layer &prevLayer){
	double sum = 0.0;

	//sum (prevVal * Weight)
	for(unsigned n = 0; n < prevLayer.size(); n++){
		sum += prevLayer[n].getOutputVal() * 
				prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
	for(unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}

// Class Net =========================================

class Net{
	public :
		Net(const vector<unsigned> &topology);
		void feedForward(const vector<double> &inputVals);
		void backProp(const vector<double> &targetVals);
		void getResults(vector<double> &resultVals) const;
		double getRecentAverageError(void) const { return m_recentAverageError; }

	private:
		vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;
};

void Net::getResults(vector<double> &resultVals) const{
	resultVals.clear();

	for(unsigned n = 0; n < m_layers.back().size() - 1; n++){
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const vector<double> &targetVals) {
	//Calculate overall net Error (RMS)
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for(unsigned n = 0; n < outputLayer.size() - 1; n++){
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error); // Root mean squared error

	// recent average error
	m_recentAverageError = 
				(m_recentAverageError * m_recentAverageSmoothingFactor + m_error )
				/ (m_recentAverageSmoothingFactor + 1.0);

	//Calculate output layer gradients
	for(unsigned n = 0; n < outputLayer.size() - 1; n++){
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Calculate gradients on hidden layer
	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--){
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for(unsigned n = 0; n < hiddenLayer.size(); ++n){
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// for all layers from output to first hidden layer,
	// update connection weights
	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n){
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals) {
	assert(inputVals.size() == m_layers[0].size() - 1);

	//assign the input values to input neuron
	for(unsigned i = 0; i < inputVals.size(); i++){
		m_layers[0][i].setOutputVal( inputVals[i] );
	}

	//Forward propagation
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		Layer &prevLayer = m_layers[layerNum - 1];
		for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology){
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; layerNum++){
		m_layers.push_back(Layer());
		unsigned numOutputs;
		if(layerNum == topology.size()-1){
			numOutputs =  0;
		}
		else {
			numOutputs = topology[layerNum + 1];
		}

		//fill it ith neurons and add bias to layer
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout<< "Made a Neuron" <<endl;
		}

		//bias node = 1.0
		m_layers.back().back().setOutputVal(1.0);
	}
}

void showVectorVals( string label, vector<double> &v){
	cout<< label << " ";
	for(unsigned i = 0; i < v.size(); i++){
		cout<< v[i] << " ";
	}
	cout<<endl;
}
		
int main(){
	TrainingData trainData("/tmp/trainingData.txt");

	// misal {3,2,3}
	vector <unsigned> topology;	
	trainData.getTopology(topology);
	Net myNet(topology);
	
	vector <double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	
	while(!trainData.isEof()){
		++trainingPass;
		cout << endl << "Pass " << trainingPass;

		//Get new input and feedforward:
		if(trainData.getNextInputs(inputVals) != topology[0]){
			break;
		}
		showVectorVals(": Inputs : ", inputVals);
		myNet.feedForward(inputVals);

		//collect the net's actual results:
		myNet.getResults(resultVals);
		showVectorVals(" Output : ", resultVals);

		//Train the net what the output should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals ("Targets : ", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		//Report how well the training is working
		cout<< myNet.getRecentAverageError() 
				<< myNet.getRecentAverageError() << endl;
	}
	cout<< endl << "Done" << endl;

}
