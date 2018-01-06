/**
 *  class representing linear neuron
 */
public class LinearNeuron extends Neuron {

    //parameter of exponential function used in calculating activation function
    public static final double ALPHA = 0.15;

    /**
     * @see Neuron#Neuron(Network, int, int, boolean)
     */
    public LinearNeuron(Network net, int inputs, int layer_number) {
        super(net, inputs, layer_number, false);
    }

    /**
     *  @see Neuron#activate(double)
     */
    @Override
    public double activate(double x) {
        return ALPHA*x;
    }

    /**
     * @see Neuron#activateDerivative(double)
     */
    @Override
    public double activateDerivative(double x) {
        return ALPHA;
    }

}