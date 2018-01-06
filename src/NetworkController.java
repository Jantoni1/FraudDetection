/**
 * Class performing training sessions on neural network
 */
public class NetworkController {


    private Network network;

    private int currentQuality;
    private int recentQuality;

    private double currentErrorCount;
    private double recentErrorCount;

    private int falsePositives;
    private int undetectedFrauds;

    private final double weight = 3.0;


    public NetworkController(Network network) {
        this.network = network;
    }


    /**
     * function classifies output value based on network's output value(s)
     * @param output output vector
     * @return classified value
     */
    public int interpretOutput(double [] output) {
        return output[0] >= 0.5 ? 1 : 0;
    }


    /**
     * method runs training sessions for the network until it stops giving better results
     * @param samplesPerEpoch number of input sets that should be used during training session
     * @param file name of input data file
     */
    public void teachBySecondAlgorithm(int samplesPerEpoch, String file) {
        initializeMembers();
        InputParser reader = new InputParser(); //TODO argumenty konstruktora
        do {
            resetMembers();
            reader.begin();
            reader.nextLine();
            runTrainingSession(samplesPerEpoch, reader);
            reader.begin();

        } while ((currentQuality < recentQuality) || (currentErrorCount < recentErrorCount));
    }

    /**
     * calculate number of wrong output produced by the network
     * @param inputParser input data source
     */
    private void checkValidationSetResults(InputParser inputParser) {
        while (inputParser.nextLine()) {
            double[] data = inputParser.getParameters();
            int[] answer = inputParser.getOutput();
            int classification = getClassification(data);
            if (answer[0] != classification) {
                ++currentQuality;
                if (classification == 0) {
                    ++undetectedFrauds;
                } else {
                    ++falsePositives;
                }
            }
        }
        currentErrorCount = falsePositives + undetectedFrauds * weight;
    }

    /**
     *
     * @param numberOfSamples number of input sets that should be used to train network
     * @param inputParser source of input data
     */
    private void runTrainingSession(int numberOfSamples, InputParser inputParser) {
        for (int j = 0; j < numberOfSamples; ++j) {
            double[] data = inputParser.getParameters();
            int[] answer = inputParser.getOutput();
            network.learn(data, answer);
            network.validate_learning();
        }
    }

    /**
     * initialize members used during training sessions
     */
    private void initializeMembers() {
        currentQuality = Integer.MAX_VALUE;
        recentQuality = Integer.MAX_VALUE;
        currentErrorCount = Double.MAX_VALUE;
        recentErrorCount = Double.MAX_VALUE;
        falsePositives = 0;
        undetectedFrauds = 0;
    }

    /**
     * reset members used during raining sessions
     */
    private void resetMembers() {
        falsePositives = 0;
        undetectedFrauds = 0;
        recentQuality = currentQuality;
        currentQuality = 0;
        recentErrorCount = currentErrorCount;
        currentErrorCount = 0;
    }

    /**
     * calls network to classify input data
     * @param input input data vector
     * @return classified value
     */
    public int getClassification(double [] input) {
        double [] output = network.classify(input);
        return interpretOutput(output);
    }

}