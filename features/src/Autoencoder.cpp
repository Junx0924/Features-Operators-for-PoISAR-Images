#include<iostream>
#include <opencv2/opencv.hpp>

#include "Autoencoder.h"


using namespace std;
using namespace cv;

/*
Written by : Anupama Rajkumar
Date : 25.06.2020
*/
Autoencoder::Autoencoder(int inputDim, int hiddenDim, double learningRate, double momentum) {
	m_dataDimension = inputDim;
	m_hiddenDimension = hiddenDim;
	m_learningRate = learningRate;
	m_momentum = momentum;

	m_inputValues.reserve(m_dataDimension);

	m_encoderWt = new float*[m_hiddenDimension];
	for (int cnt = 0; cnt < m_hiddenDimension; cnt++) {
		m_encoderWt[cnt] = this->random(m_dataDimension);
	}

	m_decoderWt = new float*[m_dataDimension];	
	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		m_decoderWt[cnt] = this->random(m_hiddenDimension);
	}

	m_decoderWtChanges = new float*[m_hiddenDimension];
	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		m_decoderWtChanges[cnt] = new float[m_dataDimension]();
	}
	m_updatedWt = new float*[m_hiddenDimension];
	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		m_updatedWt[cnt] = new float[m_dataDimension]();
	}

	m_encoderWtChanges = new float*[m_dataDimension];
	for (int cnt = 0; cnt < m_hiddenDimension; cnt++) {
		m_encoderWtChanges[cnt] = new float[m_hiddenDimension]();
	}


	m_inputBias = new float[m_hiddenDimension];			/*think of the dimension of the bias*/
	m_deltas = new float[m_dataDimension]();
}

float* Autoencoder::random(size_t elementSize) {
	float *result = new float[elementSize];
	for (size_t i = 0; i < elementSize; i++) {
		result[i] = ((float)rand() / (RAND_MAX));
	}
	return result;
}

Autoencoder::~Autoencoder() {

	for (auto i = 0; i < m_hiddenDimension; i++)
	{
		delete[] m_encoderWt[i];
		delete[] m_encoderWtChanges[i];
	}

	for (auto i = 0; i < m_dataDimension; i++)
	{
		delete[] m_decoderWt[i];
		delete[] m_decoderWtChanges[i];
	}
	delete[] m_deltas;
}

/*
Written by : Anupama Rajkumar
Date : 25.06.2020
*/
void Autoencoder::feedforward(vector<float>& m_hiddenValues, vector<float>& m_outputValues) {

	
	m_hiddenValues.reserve(m_hiddenDimension);
	m_outputValues.reserve(m_dataDimension);

	/*encoder - input->hidden layer*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		float total = 0.0;
		for (auto j = 0; j < m_dataDimension; j++) {
			total += m_encoderWt[i][j] * m_inputValues[j];	
		}
		/*assign this value to hiddenvalue after passing through activation
		activation function used: sigmoid*/
		/*todo - add bias*/
		m_hiddenValues.push_back(this->sigmoid(total));
	}

	/*decoder - hidden layer -> output*/
	for (auto i = 0; i < m_dataDimension; i++) {
		float total = 0.0;
		for (auto j = 0; j < m_hiddenDimension; j++) {
				total += m_decoderWt[i][j] * m_hiddenValues[j];		
		}
		/*assign this value to output after passing through activation
		activation function used: sigmoid*/
		m_outputValues.push_back(this->sigmoid(total));
	}

	/*printing encoder wt vector*/
	for (int x = 0; x < m_hiddenDimension; x++) {
		for (int y = 0; y < m_dataDimension; y++) {
			cout << m_encoderWt[x][y] << "\t";
		}
		cout << endl;
	}
}

/*
Modified by : Anupama Rajkumar
Date : 25.06.2020
*/
void Autoencoder::backpropagate(vector<float>& m_hiddenValues, vector<float>& m_outputValues) {
	/*for each output value - from outputlayer to hiddenlayer*/
	for (auto i = 0; i < m_dataDimension; i++) {
		m_deltas[i] = (m_outputValues[i] - m_inputValues[i])*this->sigmoidDerivation(m_outputValues[i]);
		for (auto j = 0; j < m_hiddenDimension; j++) {
			/*adjusting weights vector from the hidden layer to output*/
			m_decoderWtChanges[i][j] = m_deltas[i]*m_hiddenValues[j] ;
		}
	}
	
	/*from hidden layer to inputlayer*/
	for (auto i = 0; i < m_dataDimension; i++) {
		for (auto j = 0; j < m_hiddenDimension; j++) {
			m_updatedWt[i][j] = m_decoderWt[i][j] * m_deltas[i];
		}
	}

	for (auto i = 0; i < m_hiddenDimension; i++) {
		for (auto j = 0; j < m_dataDimension; j++) {
			float dActivation = this->sigmoidDerivation(m_hiddenValues[i]);
			m_encoderWtChanges[i][j] = m_updatedWt[j][i] * dActivation * m_inputValues[j];
		}
	}

	/*Adjusting the weights - encoder*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		for (auto j = 0; j < m_dataDimension; j++) {
			float weightChange = -(m_learningRate * m_momentum * m_encoderWtChanges[i][j]);
			m_encoderWt[i][j] += weightChange;
		}
	}

	/*Adjusting the weights - decoder*/
	for (auto i = 0; i < m_dataDimension; i++) {
		for (auto j = 0; j < m_hiddenDimension; j++) {
			float weightChange = -(m_learningRate * m_momentum * m_decoderWtChanges[i][j]);
			m_decoderWt[i][j] += weightChange;
		}
	}
}


float Autoencoder::sigmoid(float d) {
	float sigmoidVal;
	float den = (1.0 + exp(-d));
	sigmoidVal = 1.0 / den;
	cout << "den: " << den << " SigmoidVal: " << sigmoidVal << endl;
	return sigmoidVal;
}

float Autoencoder::sigmoidDerivation(float d) {
	return d * (1.0 - d);
}

void Autoencoder::train(vector<float>& data) {
	m_inputValues = data;
	vector<float> m_hiddenValues;
	vector<float> m_outputValues;
	this->feedforward(m_hiddenValues, m_outputValues);
	this->backpropagate(m_hiddenValues, m_outputValues);
}

void Autoencoder::test(vector<float>& data) {
	m_inputValues = data;
	vector<float> m_hiddenValues;
	vector<float> m_outputValues;
	this->feedforward(m_hiddenValues, m_outputValues);
}