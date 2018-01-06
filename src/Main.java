import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        Network network = new Network(29, 1, 3, 30);
        NetworkController controller = new NetworkController(network);

        try {
            controller.teach(600, new InputParser("bal.csv", ';', true));
            controller.test(new InputParser("ful.csv", ';', true));
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
