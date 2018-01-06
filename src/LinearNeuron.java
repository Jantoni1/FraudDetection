/**
 *  class representing linear neuron
 */
public class LinearNeuron extends Neuron {

    //parameter of exponential function used in calculating activation function
    private static final double ALPHA = 0.15;

    /**
     * @see Neuron#Neuron(Network, int, int, boolean)
     */
    public LinearNeuron(Network network, int numberOfInputs, int layerNumber) {
        super(network, numberOfInputs, layerNumber, false);
    }

    /**
     *  @see Neuron#activate(double)
     */
    @Override
    public double activate(double value) {
        return ALPHA * value;
    }

    /**
     * @see Neuron#activateDerivative(double)
     */
    @Override
    public double activateDerivative(double value) {
        return ALPHA;
    }
}