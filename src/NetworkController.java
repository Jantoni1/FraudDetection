import java.text.DecimalFormat;

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
     * @param inputParser training set container
     */
    public void teachUsingQuality(int samplesPerEpoch, InputParser inputParser) {
        initializeMembers();
        do {
            resetMembers();
            runTrainingSession(samplesPerEpoch, inputParser);
            checkValidationSetResults(inputParser);
        } while ((currentQuality < recentQuality) || (currentErrorCount < recentErrorCount));
    }

    /**
     * method runs training sessions for the network until it stops giving better results
     * @param samplesPerEpoch number of input sets that should be used during training session
     * @param epochs number of epochs
     * @param inputParser training set container
     */
    public void teachUsingEpochs(int samplesPerEpoch, int epochs, InputParser inputParser) {
        initializeMembers();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            runTrainingSession(samplesPerEpoch, inputParser);
        }
    }

    public void test(InputParser inputParser) {
        resetMembers();
        checkValidationSetResults(inputParser);
        DecimalFormat df = new DecimalFormat("#.##");
        System.out.println("Liczba false positives: " + falsePositives);
        System.out.println("Liczba undetected: " + undetectedFrauds);
        System.out.println("Trafnosc dla poprawnych transakcji: " + df.format(100.0 * (1 - falsePositives / 284315.0d)) + "%");
        System.out.println("Trafnosc dla fraudow: " + df.format(100 * (1 - undetectedFrauds / 492.0d)) + "%");
    }

    /**
     * calculate number of wrong output produced by the network
     * @param inputParser input data source
     */
    private void checkValidationSetResults(InputParser inputParser) {
        inputParser.begin();
        while (inputParser.nextLine()) {
            int classification = getClassification(inputParser.getParameters());
            if (inputParser.getOutput()[0] != classification) {
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
        inputParser.begin();
        for (int i = 0; i < numberOfSamples; ++i) {
            inputParser.nextLine();
            network.learn(inputParser.getParameters(), inputParser.getOutput());
            network.validateLearning();
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
    public int getClassification(double[] input) {
        double[] output = network.classify(input);
        return interpretOutput(output);
    }
}