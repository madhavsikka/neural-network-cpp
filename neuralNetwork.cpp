#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <random>
#include <iomanip>
using namespace std;

// ADAM parameters
const double lr = 0.001;
const double beta1 = 0.9;
const double beta2 = 0.999;
const double epsilon = 1e-8;

// 2D Vector renamed as matrix
typedef vector<vector<double> > matrix;

// Function Prototypes
// Definitions are written below main()
void catchError(string s);
bool assertNonZero(matrix& a);
void printMatrix(matrix a);
double generateNormalRandom(double sd);
double generateUniformRandom(double lo, double hi);
matrix matrixMultiply(matrix& a, matrix& b);
matrix elementWiseMultiply(matrix& a, matrix& b);
matrix matrixAdd(matrix& a, matrix& b);
matrix transposeMatrix(matrix& a);
matrix activateMatrix(matrix& a, string activationFn);
matrix differentiateMatrix(matrix& a, string activationFn);
void initializeMatrix(matrix& a, string activationFn);

struct linear {
	double output(double x) {
		return x;
	}
	double derivative(double x) {
		return 1;
	}
};

struct relu {
	double output(double x) {
		return max(0.0, x);
	}
	double derivative(double x) {
		if (x <= 0) return 0;
		return 1;
	}
};

struct sigmoidal {
	double output(double x) {
		return 1.0 / (1.0 + exp(-x));
	}
	double derivative(double x) {
		return output(x) * (1 - output(x));
	}
};

struct tanHyperbolic {
	double output(double x) {
		return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
	}
	double derivative(double x) {
		return (1.0 - output(x) * output(x));
	}
};

struct leakyRelu {
	double slope;
	leakyRelu() {
		slope = 0.1;
	}
	double output(double x) {
		if (x >= 0) return x;
		return slope * x;
	}
	double derivative(double x) {
		if (x <= 0) return slope;
		return 1;
	}
};

struct hiddenLayer {
	int prevDim, currDim;
	matrix weights, weightDerivatives, bias, biasDerivatives, in, z, out, error;
	matrix m, v, mb, vb;
	string activationFn;

	hiddenLayer(int m1 = 1, int n = 1, string inpActivationFn = "relu") {
		assert(m1 > 0 && n > 0);
		activationFn = inpActivationFn;
		prevDim = m1, currDim = n;
		weights.resize(currDim, vector<double>(prevDim));
		initializeMatrix(weights, inpActivationFn);
		weightDerivatives.resize(currDim, vector<double>(prevDim));
		bias.resize(currDim, vector<double>(1));
		initializeMatrix(bias, inpActivationFn);
		biasDerivatives.resize(currDim, vector<double>(1));
		error.resize(currDim, vector<double>(1));
		z.resize(currDim, vector<double>(1));
		in.resize(prevDim, vector<double>(1));
		out.resize(currDim, vector<double>(1));
		m.resize(currDim, vector<double>(prevDim, 0));
		v.resize(currDim, vector<double>(prevDim, 0));
		mb.resize(currDim, vector<double>(1, 0));
		vb.resize(currDim, vector<double>(1, 0));
	}

	matrix forward(matrix& input) {
		in = input;
		z = matrixMultiply(weights, in);
		z = matrixAdd(z, bias);
		out = activateMatrix(z, activationFn);
		return out;
	}

	matrix backward(matrix& weightedError) {
		assert(weightedError.size() == currDim);
		assert(weightedError[0].size() == 1);
		assertNonZero(z);
		matrix derivedMat = differentiateMatrix(z, activationFn);
		error = elementWiseMultiply(weightedError, derivedMat);
		biasDerivatives = error;
		for (int j = 0; j < currDim; j++) for (int k = 0; k < prevDim; k++) weightDerivatives[j][k] = in[k][0] * error[j][0];
		matrix weightTranspose = transposeMatrix(weights);
		matrix backWeightedError = matrixMultiply(weightTranspose, error);
		return backWeightedError;
	}

	// ADAM
	void updateWeights(int iteration) {
		for (int i = 0; i < currDim; i++) {
			mb[i][0] = beta1 * mb[i][0] + (1 - beta1) * biasDerivatives[i][0];
			vb[i][0] = beta2 * vb[i][0] + (1 - beta2) * biasDerivatives[i][0] * biasDerivatives[i][0];
			for (int j = 0; j < prevDim; j++) {
				m[i][j] = beta1 * m[i][j] + (1 - beta1) * weightDerivatives[i][j];
				v[i][j] = beta2 * v[i][j] + (1 - beta2) * weightDerivatives[i][j] * weightDerivatives[i][j];
			}
		}
		double divM = 1 - pow(beta1, iteration);
		double divV = 1 - pow(beta2, iteration);
		for (int i = 0; i < currDim; i++) {
			bias[i][0] -= (lr / (sqrt(vb[i][0] / divV) + epsilon) * (mb[i][0] / divM));
			for (int j = 0; j < prevDim; j++) weights[i][j] -= (lr / (sqrt(v[i][j] / divV) + epsilon) * (m[i][j] / divM));
		}
	}
};

struct inputLayer {
	int currDim;
	matrix in;

	inputLayer(int n = 0) {
		currDim = n;
	}

	matrix forward(matrix& input) {
		in = input;
		return in;
	}
};

struct neuralNet {
	int inputDim, numHiddenLayers, outputDim;
	double loss;
	vector<string> activationFns;
	inputLayer il;
	vector<hiddenLayer> hl;
	hiddenLayer ol;

	neuralNet(int numInputNeurons, vector<int> numHiddenNeurons, int numOutputNeurons, vector<string>& inpActivationFn) {
		activationFns = inpActivationFn;
		inputDim = numInputNeurons, outputDim = numOutputNeurons, numHiddenLayers = numHiddenNeurons.size();
		il = inputLayer(inputDim);
		hl.resize(numHiddenLayers);
		int prev = numInputNeurons;
		for (int i = 0; i < numHiddenLayers; i++) {
			hl[i] = hiddenLayer(prev, numHiddenNeurons[i], activationFns[i]);
			prev = numHiddenNeurons[i];
		}
		ol = hiddenLayer(prev, numOutputNeurons, activationFns.back());
	}

	matrix forward(vector<double>& xi) {
		matrix x = matrix(1, xi);
		x = transposeMatrix(x);
		matrix curr = il.forward(x);
		for (int i = 0; i < hl.size(); i++) curr = hl[i].forward(curr);
		curr = ol.forward(curr);
		return curr;
	}

	void calcLoss(vector<double>& yi) {
		int n = yi.size();
		assert(n == ol.out.size());
		double currLoss = 0;
		for (int i = 0; i < n; i++) currLoss += (yi[i] - ol.out[i][0]) * (yi[i] - ol.out[i][0]);
		currLoss /= 2;
		loss = currLoss;
	}

	void backward(vector<double>& y) {
		int n = y.size();
		assert(n == ol.out.size());

		matrix gradC(n, vector<double>(1));
		for (int i = 0; i < n; i++) gradC[i][0] = ol.out[i][0] - y[i];
		assertNonZero(gradC);

		matrix currWeightedError = ol.backward(gradC);
		for (int i = numHiddenLayers - 1; i >= 0; i--) currWeightedError = hl[i].backward(currWeightedError);
	}

	void updateWeights(int iteration) {
		for (int i = 0; i < hl.size(); i++) hl[i].updateWeights(iteration);
		ol.updateWeights(iteration);
	}

	matrix predict(vector<double>& x) {
		return forward(x);
	}

	double getLoss() {
		return loss;
	}
};

int main() {

	std::cout << std::fixed << std::showpoint;
	std::cout << std::setprecision(6);

	ifstream infile;
	infile.open("train_data.txt");
	if (!infile) {
		cout << "Error in opening file.";
		return 0;
	}

	// Input Data
	int inputDim, outputDim, numHiddenLayers;
	vector<int> numHiddenNeurons;
	infile >> inputDim >> outputDim;
	infile >> numHiddenLayers;
	numHiddenNeurons.resize(numHiddenLayers);
	for (int i = 0; i < numHiddenLayers; i++) infile >> numHiddenNeurons[i];
	vector<string> activationFns(numHiddenLayers + 1);
	for (int i = 0; i <= numHiddenLayers; i++) infile >> activationFns[i];


	int numEpochs = 1;
	infile >> numEpochs;

	// Train-Test Split
	int numDataPoints, numTrain, numTest;
	infile >> numDataPoints;
	numTrain = 0.9 * numDataPoints;
	numTest = numDataPoints - numTrain;

	matrix xTrain, yTrain, xTest, yTest;
	xTrain.resize(numTrain, vector<double>(inputDim));
	yTrain.resize(numTrain, vector<double>(outputDim));
	xTest.resize(numTest, vector<double>(inputDim));
	yTest.resize(numTest, vector<double>(outputDim));


	for (int i = 0; i < numTrain; i++) {
		for (int j = 0; j < inputDim; j++) infile >> xTrain[i][j];
		for (int j = 0; j < outputDim; j++) infile >> yTrain[i][j];
	}
	for (int i = 0; i < numTest; i++) {
		for (int j = 0; j < inputDim; j++) infile >> xTest[i][j];
		for (int j = 0; j < outputDim; j++) infile >> yTest[i][j];
	}


	// Initialize Neural Network
	neuralNet nn(inputDim, numHiddenNeurons, outputDim, activationFns);

	int maxEpochs = 150;
	double trainLoss = 0, testLoss;

	for (int epoch = 1; epoch <= numEpochs; epoch++) {
		double avgLoss = 0;
		for (int i = 0; i < numTrain; i++) {
			nn.forward(xTrain[i]);
			nn.calcLoss(yTrain[i]);
			nn.backward(yTrain[i]);
			nn.updateWeights(epoch);
			avgLoss += nn.getLoss();
		}
		avgLoss /= numTrain;
		trainLoss = avgLoss;
		cout << "Epoch " << epoch << ", MSE Loss = " << avgLoss << endl;
	}

	for (int i = 0; i < numTest; i++) {
		nn.predict(xTest[i]);
		nn.calcLoss(yTest[i]);
		testLoss += nn.getLoss();
	}
	testLoss /= numTest;

	cout << "\n=================================\n";
	cout << "      Model Trained (90% Data)";
	cout << "\n=================================\n";
	cout << "    Train MSE Loss = " << trainLoss;
	cout << "\n=================================\n";
	cout << " Running on Test Data (10% Data) ";
	cout << "\n=================================\n";
	cout << "     Test MSE Loss = " << testLoss;
	cout << "\n=================================\n\n";
	cout << "\n#################################\n\n";

	infile.close();

	while (1) {
		int temp;
		cout << "- Press 1 to make predictions\n- Press 2 to exit\n";
		cin >> temp;
		if (temp == 2) return 0;
		cout << "Enter data point (" << inputDim << " Attributes): \n";
		vector<double> xpred(inputDim);
		for (int i = 0; i < inputDim; i++) cin >> xpred[i];
		cout << "Predicted Output:\n";
		printMatrix(nn.predict(xpred));
		cout << "\n#################################\n";
	}

	return 0;

}

// ==========================================================================================
// Function Definitions
// ==========================================================================================


void catchError(string s) {
	cout << s << endl;
	exit(1);
}

bool assertNonZero(matrix& a) {
	int n = a.size();
	assert(n != 0);
	int m = a[0].size();
	assert(m != 0);
	bool flag = true;
	for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) if (a[i][j] != 0) flag = false;
	// if (flag) catchError("Zero Matrix");
	// if (flag) cout << "Zero Matrix\n";
	return flag;
}

void printMatrix(matrix a) {
	int n = a.size();
	assert(n != 0);
	int m = a[0].size();
	assert(m != 0);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) cout << a[i][j] << " ";
		cout << "| ";
	}
	cout << "\n";
}

double generateNormalRandom(double sd) {
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::normal_distribution<double> distr(0.0, sd);
	return distr(generator);
}

double generateUniformRandom(double lo, double hi) {
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<> distr(lo, hi);
	return distr(generator);
}

matrix matrixMultiply(matrix& a, matrix& b) {
	int n1 = a.size(), m1 = a[0].size(), n2 = b.size(), m2 = b[0].size();
	assert(m1 == n2);
	matrix res(n1, vector<double>(m2));
	for (int i = 0; i < n1; i++) {
		for (int j = 0; j < m2; j++) {
			for (int k = 0; k < m1; k++)
				res[i][j] += a[i][k] * b[k][j];
		}
	}
	return res;
}

matrix elementWiseMultiply(matrix& a, matrix& b) {
	assert(a.size() == b.size() && a[0].size() == b[0].size());
	int n = a.size(), m = a[0].size();
	matrix res(n, vector<double>(m));
	for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = a[i][j] * b[i][j];
	return res;
}

matrix matrixAdd(matrix& a, matrix& b) {
	assert(a.size() == b.size());
	assert(a[0].size() == b[0].size());
	int n = a.size(), m = a[0].size();
	matrix res(n, vector<double>(m));
	for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = a[i][j] + b[i][j];
	return res;
}

matrix transposeMatrix(matrix& a) {
	int n = a.size(), m = a[0].size();
	matrix res = matrix(m, vector<double>(n));
	for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) res[i][j] = a[j][i];
	return res;
}

matrix activateMatrix(matrix& a, string activationFn) {
	int n = a.size(), m = a[0].size();
	matrix res(n, vector<double>(m));
	if (activationFn == "relu") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = relu().output(a[i][j]);
	else if (activationFn == "sigmoidal") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = sigmoidal().output(a[i][j]);
	else if (activationFn == "tanHyperbolic") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = tanHyperbolic().output(a[i][j]);
	else if (activationFn == "leakyRelu") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = leakyRelu().output(a[i][j]);
	else if (activationFn == "linear") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = linear().output(a[i][j]);
	else catchError("Invalid Activation");
	return res;
}

matrix differentiateMatrix(matrix& a, string activationFn) {
	int n = a.size();
	assert(n != 0);
	int m = a[0].size();
	matrix res(n, vector<double>(m));
	if (activationFn == "relu") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = relu().derivative(a[i][j]);
	else if (activationFn == "sigmoidal") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = sigmoidal().derivative(a[i][j]);
	else if (activationFn == "tanHyperbolic") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = tanHyperbolic().derivative(a[i][j]);
	else if (activationFn == "leakyRelu") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = leakyRelu().derivative(a[i][j]);
	else if (activationFn == "linear") for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) res[i][j] = linear().derivative(a[i][j]);
	else catchError("Invalid Activation");
	return res;
}

void initializeMatrix(matrix& a, string activationFn) {
	int n = a.size();
	assert(n != 0);
	int m = a[0].size();
	assert(m != 0);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (activationFn == "leakyRelu" || activationFn == "relu") a[i][j] = generateNormalRandom(sqrt(2.0 / m));
			else if (activationFn == "tanHyperbolic") a[i][j] = generateNormalRandom(sqrt(1.0 / m));
			else a[i][j] = generateUniformRandom(-1.0 / sqrt(m), 1.0 / sqrt(m));
		}
	}
}

