import java.lang.Math;


/**
 *  class representing a sigmoidal bipolar neuron
 */
public class SigmoidalNeuron extends Neuron {

    //parameter of linear function used to calculate neuron's activate function
    public static final double ALPHA = 1.0;

    /**
     * @see Neuron#Neuron(Network, int, int, boolean)  
     */
    public SigmoidalNeuron(Network net, int inputs, int layer_number) {
        super(net, inputs, layer_number, true);
    }

    /**
     * @see Neuron#activate(double) 
     */
    @Override
    public double activate(double x) {
        return (1.0 - Math.exp(-ALPHA*x))/(1.0 + Math.exp(-ALPHA*x));
    }

    /**
     * @see Neuron#activateDerivative(double) 
     */
    @Override
    public double activateDerivative(double x) {
        return ALPHA*(1 - output*output);
    }

}