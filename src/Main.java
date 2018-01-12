import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        Network network = new Network(29, 1, 4, 40);
        NetworkController controller = new NetworkController(network);
        try {
            //controller.teachUsingQuality(400, new InputParser("bal.csv", ';', true));
            controller.teachUsingEpochs(400, 100, new InputParser("bal.csv", ';', true));
            controller.test(new InputParser("ful.csv", ';', true));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
