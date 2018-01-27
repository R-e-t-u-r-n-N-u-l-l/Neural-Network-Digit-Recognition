#include "olcConsoleGameEngine.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <time.h>
#include <string>
#include <cmath>

class Neuron;
typedef std::vector<Neuron> Layer;

class File {
	
private:
	int MAX_ITERATIONS, DATA_SIZE;
	std::vector<unsigned> layout;
	std::vector< std::vector<double> > inputs;
	std::vector< std::vector<double> > targets;

public:
	File(const char* filePath);
	std::vector<double> getInputs(const int index) const { return inputs[index]; };
	std::vector<double> getTargets(const int index) const { return targets[index]; };
	std::vector<unsigned> getLayout() const { return layout; };
	int getMaxIterations() const { return MAX_ITERATIONS; };
	int getDataSize() const { return DATA_SIZE; };

};

std::vector<std::string> split(std::string str, char c) {
	std::vector<std::string> array;
	std::string element = "";

	for (unsigned i = 0; i < str.length(); i++) {
		if (str[i] != c)
			element += str[i];
		else if (str[i] == c && element != "") {
			array.push_back(element);
			element = "";
		}
	} if (element != "")
		array.push_back(element);

	return array;
}

File::File(const char* filePath) {
	std::string line;
	std::vector<std::string> part;
	std::ifstream file(filePath);

	if (file.is_open()) {
		int index = 0;
		while (std::getline(file, line)) {
			if (index == 0)
				MAX_ITERATIONS = atoi(line.c_str());
			else if (index == 1) {
				part = split(line, ' ');
				for (unsigned p = 0; p < part.size(); p++)
					layout.push_back(atoi(part[p].c_str()));
			}
			else if (index % 2 == 0) {
				std::vector<double> i;
				part = split(line, ' ');
				for (unsigned p = 0; p < part.size(); p++)
					i.push_back(atof(part[p].c_str()));
				inputs.push_back(i);
			}
			else {
				std::vector<double> t;
				part = split(line, ' ');
				for (unsigned p = 0; p < part.size(); p++)
					t.push_back(atof(part[p].c_str()));
				targets.push_back(t);
			}
			index++;
		}
	}
	DATA_SIZE = inputs.size();
	file.close();
}

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {

private:
	static double learningRate;
	static double alpha;
	//static double activate(double value) { return tanh(value); }
	static double activate(double value) { return 1 / (1 + exp(-value)); }
	//static double activateDerivative(double value) { return 1 - tanh(value) * tanh(value); }
	static double activateDerivative(double value) { return activate(value) * (1 - activate(value)); }
	static double random(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double output;
	std::vector<Connection> outputWeights;
	unsigned index;
	double gradient;

public:
	Neuron(unsigned outputAmt, unsigned index);
	void setOutput(double value) { output = value; }
	double getOutput(void) const { return output; }
	std::vector<Connection> getOutputWeights() const { return outputWeights; }
	void feedForward(const Layer &prevLayer);
	void calculateOutputGradients(double target);
	void calculateHiddenGradients(const Layer &nextLayer);
	void updateWeights(Layer &prevLayer);

};

double Neuron::learningRate = 0.2;
double Neuron::alpha = 0.5;


void Neuron::updateWeights(Layer &prevLayer) {
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		double oldDeltaWeight = prevLayer[n].outputWeights[index].deltaWeight;

		double newDeltaWeight = learningRate * prevLayer[n].getOutput() * gradient + alpha * oldDeltaWeight;

		prevLayer[n].outputWeights[index].deltaWeight = newDeltaWeight;
		prevLayer[n].outputWeights[index].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
		sum += outputWeights[n].weight * nextLayer[n].gradient;

	return sum;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::activateDerivative(output);
}

void Neuron::calculateOutputGradients(double target) {
	double delta = target - output;
	gradient = delta * Neuron::activateDerivative(output);
}

void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); n++)
		sum += prevLayer[n].getOutput() * prevLayer[n].outputWeights[index].weight;

	output = Neuron::activate(sum);
}

Neuron::Neuron(unsigned outputAmt, unsigned index) {
	this->index = index;
	outputWeights.reserve(outputAmt);

	for (unsigned i = 0; i < outputAmt; i++) {
		outputWeights.push_back(Connection());
		outputWeights.back().weight = random();
	}
}


class Network {

private:
	std::vector<Layer> layers;
	double error;
	double averageError;
	static double smoothingFactor;

public:
	Network(const std::vector<unsigned> &layout);
	void feedForward(const std::vector<double> &inputs);
	void backProp(const std::vector<double> &targets);
	void getResults(std::vector<double> &results) const;
	double getRecentAverageError(void) const { return averageError; }
	std::vector<Layer> getLayers() const { return layers; }

};


double Network::smoothingFactor = 100;


void Network::getResults(std::vector<double> &results) const {
	results.clear();
	results.reserve(layers.back().size());

	for (unsigned n = 0; n < layers.back().size() - 1; n++) {
		results.push_back(layers.back()[n].getOutput());
	}
}

void Network::backProp(const std::vector<double> &targets) {
	Layer &outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		double delta = targets[n] - outputLayer[n].getOutput();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1;
	error = sqrt(error);

	averageError = (averageError * smoothingFactor + error) / (smoothingFactor + 1.0);

	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		outputLayer[n].calculateOutputGradients(targets[n]);
	}

	for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
		for (unsigned n = 0; n < layers[layerNum].size(); n++) {
			layers[layerNum][n].calculateHiddenGradients(layers[layerNum + 1]);
		}
	}

	for (unsigned layerNum = layers.size() - 1; layerNum > 0; layerNum--) {
		for (unsigned n = 0; n < layers[layerNum].size() - 1; n++) {
			layers[layerNum][n].updateWeights(layers[layerNum - 1]);
		}
	}
}

void Network::feedForward(const std::vector<double> &inputVals) {
	for (unsigned i = 0; i < inputVals.size(); i++) {
		layers[0][i].setOutput(inputVals[i]);
	}

	for (unsigned layerNum = 1; layerNum < layers.size(); layerNum++) {
		for (unsigned n = 0; n < layers[layerNum].size() - 1; n++)
			layers[layerNum][n].feedForward(layers[layerNum - 1]);
	}
}

Network::Network(const std::vector<unsigned> &layout) {
	layers.reserve(layout.size());
	error = 0;
	averageError = 0;
	for (unsigned l = 0; l < layout.size(); l++) {
		layers.push_back(Layer());
		unsigned outputAmt = l == layout.size() - 1 ? 0 : layout[l + 1];

		layers.back().reserve(layout[l]);
		for (unsigned n = 0; n <= layout[l]; n++)
			layers.back().push_back(Neuron(outputAmt, n));

		layers.back().back().setOutput(1);
	}
}

class Graphics : public olcConsoleGameEngine {

private:
	Network & net;
	const int gridSize = 32;
	int guess = 0;
	bool pixelState[32 * 32];

	void drawGrid() {
		for (int i = 0; i < gridSize; i++) {
			for (int j = 0; j < gridSize; j++) {
				if (pixelState[j * gridSize + i])
					Draw(ScreenWidth() / 2 - gridSize / 2 + i, ScreenHeight() / 2 - gridSize / 2 + j, PIXEL_SOLID, FG_WHITE);
				else
					Draw(ScreenWidth() / 2 - gridSize / 2 + i, ScreenHeight() / 2 - gridSize / 2 + j, PIXEL_SOLID, FG_DARK_GREY);
			}
		}
	}

public:
	Graphics(Network& n) :net(n) {
	}

	virtual bool OnUserCreate() {
		for (int i = 0; i < gridSize * gridSize; i++)
			pixelState[i] = 0;

		return true;
	}

	virtual bool OnUserUpdate(float eTime) {
		int location = (m_mousePosY - (ScreenHeight() / 2 - gridSize / 2)) * gridSize + (m_mousePosX - (ScreenWidth() / 2 - gridSize / 2));
		if (m_mouse[0].bHeld && location > 0 && location < 32 * 32)
			pixelState[location] = 1;

		DrawString(ScreenWidth() / 2 - 6, 1, L"CLEAR");
		DrawString(ScreenWidth() / 2 + 1, 1, L"SUBMIT");

		if (m_mouse[0].bReleased) {
			if (m_mousePosX >= ScreenWidth() / 2 - 6 && m_mousePosX < ScreenWidth() / 2 - 1 && m_mousePosY == 1) {
				for (int i = 0; i < gridSize * gridSize; i++)
					pixelState[i] = 0;
			} else if (m_mousePosX >= ScreenWidth() / 2 + 1 && m_mousePosX < ScreenWidth() / 2 + 7 && m_mousePosY == 1) {
				std::vector<double> inputs;
				for (int i = 0; i < gridSize * gridSize; i++)
					inputs.push_back(pixelState[i]);

				net.feedForward(inputs);

				std::vector<double> results;
				net.getResults(results);

				guess = 0;
				for (unsigned i = 1; i < results.size(); i++) {
					if (results[i] > results[guess])
						guess = i;
				}
			}
		}

		drawGrid();

		DrawString(ScreenWidth() / 2 + gridSize / 2 + (ScreenWidth() / 2 - gridSize / 2) / 2 - 3, ScreenHeight() / 2, L"GUESS: " + to_wstring(guess));

		return true;
	}

};

int main() {
	File file("data.txt");
	srand(time(NULL));

	Network net(file.getLayout());

	Graphics graphics(net);

	int iteration = 0;

	while (iteration < file.getMaxIterations()) {
		std::cout << "Iteration: " << iteration << std::endl;

		std::vector<double> inputs = file.getInputs(iteration % file.getDataSize());
		net.feedForward(inputs);

		std::cout << "Inputs: " << std::flush;

		for (unsigned i = 0; i < inputs.size(); i++)
			std::cout << inputs[i] << " " << std::flush;

		std::vector<double> targets = file.getTargets(iteration % file.getDataSize());
		net.backProp(targets);

		std::cout << std::endl << "Targets: " << std::flush;

		for (unsigned i = 0; i < targets.size(); i++)
			std::cout << targets[i] << " " << std::flush;

		std::cout << std::endl << "Results: " << std::flush;

		std::vector<double> results;
		net.getResults(results);

		for (unsigned i = 0; i < results.size(); i++)
			std::cout << results[i] << " " << std::flush;

		std::cout << std::endl << "Average recent error: " << net.getRecentAverageError() << std::endl << std::endl;
		iteration++;

		if (iteration == file.getMaxIterations()) {
			std::string choice;
			std::cout << "Neural network has reached expected amount of iterations." << std::endl;
			std::cout << "Enter Y to test the neural network, enter N to keep training it" << std::endl;
			std::cin >> choice;
			std::cin.ignore(256, '\n');
			if (choice == "Y" || choice == "y") {
				graphics.ConstructConsole(68, 40, 22, 22);
				graphics.Start();
			}
			iteration = 0;
		}
	}
}