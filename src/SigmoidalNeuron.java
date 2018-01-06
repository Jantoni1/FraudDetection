import java.lang.Math;

/**
 *  class representing a sigmoidal bipolar neuron
 */
public class SigmoidalNeuron extends Neuron {

    //parameter of linear function used to calculate neuron's activate function
    private static final double ALPHA = 1.0;

    /**
     * @see Neuron#Neuron(Network, int, int, boolean)  
     */
    public SigmoidalNeuron(Network network, int numberOfInputs, int layerNumber) {
        super(network, numberOfInputs, layerNumber, true);
    }

    /**
     * @see Neuron#activate(double) 
     */
    @Override
    public double activate(double value) {
        return (1.0 - Math.exp(-ALPHA * value)) / (1.0 + Math.exp(-ALPHA * value));
    }

    /**
     * @see Neuron#activateDerivative(double) 
     */
    @Override
    public double activateDerivative(double x) {
        return ALPHA * (1 - output * output);
    }
}