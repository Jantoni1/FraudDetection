public class Network {
    private static final double DEFAULT_LEARNING_RATE = 0.1;

    private int numberOfInputs;
    private int numberOfOutputs;
    // number of the output (last) layer
    private int outputLayer;
    // number of neurons in each hidden layer
    private int numberOfNeuronsPerLayer;

    private double learningRate;
    private double[] input;

    private double[] learnVector;
    private Neuron[][] neurons;
    private int learningIteration;

    /**
     *
     * @param inputs number of input values
     * @param outputs number of network's output values
     * @param hiddenLayers number of hidden layers
     * @param neuronsPerLayer number of neurons per each layer
     */
    public Network(int inputs, int outputs, int hiddenLayers, int neuronsPerLayer) {
        //initializeMembers(inputs, outputs, hiddenLayers, neuronsPerLayer);
        this.numberOfInputs = inputs;
        this.numberOfOutputs = outputs;
        this.outputLayer = hiddenLayers;
        this.numberOfNeuronsPerLayer = neuronsPerLayer;
        this.input = new double[numberOfInputs];
        this.learnVector = new double[numberOfOutputs];
        this.learningIteration = 0;

        // create empty layers
        neurons = new Neuron[hiddenLayers + 1][];
        // fill layers with neurons
        createLayers();
        // init network
        init(DEFAULT_LEARNING_RATE);
    }

    private void createLayers() {
        // 1. create first layer
        neurons[0] = new Neuron[numberOfNeuronsPerLayer];
        for (int i = 0; i < numberOfNeuronsPerLayer; ++i) {
            neurons[0][i] = new SigmoidalNeuron(this, numberOfInputs, 0);
        }
        // 2. create following hidden layers
        for (int i = 1; i < outputLayer; ++i) {
            neurons[i] = new Neuron[numberOfNeuronsPerLayer];
            for (int j = 0; j < numberOfNeuronsPerLayer; ++j) {
                neurons[i][j] = new SigmoidalNeuron(this, numberOfNeuronsPerLayer, i);
            }
        }
        // 3. Create output layer
        neurons[outputLayer] = new Neuron[numberOfOutputs];
        for (int i = 0; i < numberOfOutputs; ++i) {
            neurons[outputLayer][i] = new LinearNeuron(this, numberOfNeuronsPerLayer, outputLayer);
        }

    }

    public int getNumberOfLayers() {
        return outputLayer;
    }

    public int getNumberOfOutputs() {
        return numberOfOutputs;
    }

    public Neuron[][] getNeurons() {
        return neurons;
    }

    public double[] getInput() {
        return input;
    }

    public double[] getLearnVector() {
        return learnVector;
    }

    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * @param layer number of layer in network
     * @return number of neurons in this particular layer
     */
    public int getLayersSize(int layer) {
        return (layer == outputLayer - 1) ? numberOfOutputs : numberOfNeuronsPerLayer;
    }

    /**
     * initialize neuron network
     * @param learningFactor learn tempo of network
     */
    private void init(double learningFactor) {
        learningRate = learningFactor;
        for (int i = 0; i < outputLayer; ++i) {
            for (int j = 0; j < numberOfNeuronsPerLayer; ++j) {
                neurons[i][j].setNewWeights(false);
            }
        }
        for (int j = 0; j < numberOfOutputs; ++j) {
            neurons[outputLayer][j].setNewWeights(true);
        }
        learningIteration = 0;
    }

    /**
     * function works properly after network passes training
     * @param inputVector input values that will be processed by network
     * @return output values calculated by the network
     */
    public double[] classify(double[] inputVector) {
        // Copy input vector to network's input layer
        for (int i = 0; i < numberOfInputs; ++i) {
            input[i] = inputVector[i];
        }
        // Calculate network's output
        for (int i = 0; i < outputLayer; ++i)
            for(int j = 0; j < numberOfNeuronsPerLayer; ++j)
                neurons[i][j].calculateOutput();
        for (int i = 0; i < numberOfOutputs; ++i)
            neurons[outputLayer][i].calculateOutput();
        // Copy answer from network's output to output vector
        double[] output  = new double[numberOfOutputs];
        for (int i = 0; i < numberOfOutputs; ++i) {
            output[i] = neurons[outputLayer][i].getOutput();
        }
        return output;
    }


    /**
     * function that trains network to classify input as accurately as possible
     * it calculates corrections of weights
     * @param inputVector input values given to the network
     * @param learningPattern vector of correct output values that will be used to calculate error
     */
    void learn(double[] inputVector, int[] learningPattern) {
        // calculate network's output
        double[] output = classify(inputVector);
        // copy correct output vector to class member, so that it will be accessible from neurons
        for (int i = 0; i < numberOfOutputs; ++i) {
            learnVector[i] = learningPattern[i];
        }

        // calculate new weights in the network
        // here backprop algorithm is used, we calculate output corrections first
        for (int i = 0; i < numberOfOutputs; ++i) {
            neurons[outputLayer][i].calculateCorrections(i);
        }
        // then we calculate corrections to all layers weights starting from the one closest to output layer
        for (int i = outputLayer - 1; i >= 0; --i)
            for (int j = 0; j < numberOfNeuronsPerLayer; ++j) {
                neurons[i][j].calculateCorrections(j);
            }
        ++learningIteration;
    }

    /**
     * Update weights values with previously calculated changes
     */
    public void validateLearning() {
        // update values
        for (int i = 0; i < numberOfOutputs; ++i) {
            neurons[outputLayer][i].correctWeights(learningIteration);
        }
        for (int i = outputLayer - 1; i >= 0; --i)
            for (int j = 0; j < numberOfNeuronsPerLayer; ++j) {
                neurons[i][j].correctWeights(learningIteration);
            }
        learningIteration = 0;
    }
}