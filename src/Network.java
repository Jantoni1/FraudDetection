public class Network {
    public static final double DEFAULT_LEARNING_RATE = 0.1;


    private int NUMBER_OF_INPUTS;
    private int NUMBER_OF_OUTPUTS;
    // number of the output (last) layer
    private int OUTPUT_LAYER;
    // number of neurons in each hidden layer
    private int NUMBER_OF_NEURONS_EACH_LAYER;

    double learning_rate;
    double[] input;
    double[] learnVector;
    Neuron[][] neuron;
    private int learning_iteration;

    /**
     *
     * @param inputs number of input values
     * @param outputs number of network's output values
     * @param hiddenLayers number of hidden layers
     * @param neuronsPerLayer number of neurons per each layer
     */
    Network(int inputs, int outputs, int hiddenLayers, int neuronsPerLayer) {
        initializeMembers(inputs, outputs, hiddenLayers, neuronsPerLayer);

        // create empty layers
        neuron = new Neuron[hiddenLayers + 1][];
        // fill layers with neurons
        createLayers();

    }

    public void createLayers() {
        // 1. create first layer
        neuron[0] = new Neuron[NUMBER_OF_NEURONS_EACH_LAYER];
        for (int i = 0; i < NUMBER_OF_NEURONS_EACH_LAYER; ++i) {
            neuron[0][i] = new SigmoidalNeuron(this, NUMBER_OF_INPUTS, 0);
        }
        // 2. create following hidden layers
        for (int i = 1; i < OUTPUT_LAYER; ++i) {
            neuron[i] = new Neuron[NUMBER_OF_NEURONS_EACH_LAYER];
            for (int j = 0; j < NUMBER_OF_NEURONS_EACH_LAYER; ++j) {
                neuron[i][j] = new SigmoidalNeuron(this, NUMBER_OF_NEURONS_EACH_LAYER, i);
            }
        }
        // 3. Create output layer
        neuron[OUTPUT_LAYER] = new Neuron[NUMBER_OF_OUTPUTS];
        for (int i = 0; i < NUMBER_OF_OUTPUTS; ++i) {
            neuron[OUTPUT_LAYER][i] = new LinearNeuron(this, NUMBER_OF_NEURONS_EACH_LAYER, OUTPUT_LAYER);
        }

    }

    public int getNumberOfLayers() {
        return OUTPUT_LAYER;
    }
    
    /**
     * @param layer number of layer in network
     * @return number of neurons in this particular layer
     */
    public int getLayersSize(int layer) {
        return (layer == OUTPUT_LAYER - 1) ? NUMBER_OF_OUTPUTS : NUMBER_OF_NEURONS_EACH_LAYER;
    }

    /**
     * assign given values to network's parameters
     * @see Network#Network(int, int, int, int)
     */
    private void initializeMembers(int inputs, int outputs, int hiddenLayers, int neuronsPerLayer) {
        learning_rate = DEFAULT_LEARNING_RATE;
        NUMBER_OF_INPUTS = inputs;
        NUMBER_OF_OUTPUTS = outputs;
        OUTPUT_LAYER = hiddenLayers;
        NUMBER_OF_NEURONS_EACH_LAYER = neuronsPerLayer;
        input = new double[NUMBER_OF_INPUTS];
        learnVector = new double[NUMBER_OF_OUTPUTS];
        learning_iteration = 0;
    }

    
    /**
     * initialize neuron network
     * @param learningFactor learn tempo of network
     */
    void init(double learningFactor) {
        learning_rate = learningFactor;
        for(int i = 0; i < OUTPUT_LAYER; ++i) {
            for(int j = 0; j < NUMBER_OF_NEURONS_EACH_LAYER; ++j) {
                neuron[i][j].setNewWeights(false);
            }
        }
        for(int j = 0; j < NUMBER_OF_OUTPUTS; ++j) {
            neuron[OUTPUT_LAYER][j].setNewWeights(true);
        }
        learning_iteration = 0;
    }


    /**
     *  function works properly after network passes training
     * @param inputVector input values that will be processed by network
     * @return output values calculated by the network
     */
    double [] classify(double [] inputVector) {
        // Copy input vector to network's input layer
        for(int i = 0; i < NUMBER_OF_INPUTS; ++i) {
            input[i] = inputVector[i];
        }
        // Calculate network's output
        for(int i = 0; i < OUTPUT_LAYER; ++i)
            for(int j = 0; j < NUMBER_OF_NEURONS_EACH_LAYER; ++j)
                neuron[i][j].calculateOutput();
        for(int i = 0; i < NUMBER_OF_OUTPUTS; ++i)
            neuron[OUTPUT_LAYER][i].calculateOutput();
        // Copy answer from network's output to output vector
        double [] output  = new double[NUMBER_OF_OUTPUTS];
        for(int i = 0; i < NUMBER_OF_OUTPUTS; ++i) {
            output[i] = neuron[OUTPUT_LAYER][i].output;
        }
        return output;
    }


    /**
     *  function that trains network to classify input as accurately as possible
     * it calculates corrections of weights
     * @param inputVector input values given to the network
     * @param learningPattern vector of correct output values that will be used to calculate error
     * @return output vector calculated by the network
     */
    double [] learn(double [] inputVector, double [] learningPattern) {
        // calculate network's output
        double [] output = classify(inputVector);
        // copy correct output vector to class member, so that it will be accessible from neurons
        for(int i = 0; i < NUMBER_OF_OUTPUTS; ++i) {
            learnVector[i] = learningPattern[i];
        }

        // calculate new weights in the network
        // here backprop algorithm is used, we calculate output corrections first
        for(int i = 0; i < NUMBER_OF_OUTPUTS; ++i) {
            neuron[OUTPUT_LAYER][i].calculateCorrections(i);
        }
        // then we calculate corrections to all layers weights starting from the one closest to output layer
        for(int i = OUTPUT_LAYER - 1; i >= 0; --i)
            for(int j = 0; j < NUMBER_OF_NEURONS_EACH_LAYER; ++j) {
                neuron[i][j].calculateCorrections(j);
            }
        ++learning_iteration;
        return output;
    }

    
    /**
     * Update weights values with previously calculated changes
     */
    void validate_learning() {
        //update values
        for(int i = 0; i < NUMBER_OF_OUTPUTS; ++i) {
            neuron[OUTPUT_LAYER][i].correctWeights(i, learning_iteration);
        }
        for(int i = OUTPUT_LAYER - 1; i >= 0; --i)
            for(int j = 0; j < NUMBER_OF_NEURONS_EACH_LAYER; ++j) {
                neuron[i][j].correctWeights(j, learning_iteration);
            }
        learning_iteration = 0;
    }

}