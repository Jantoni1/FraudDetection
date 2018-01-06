import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class InputParser {

    private List<String> data;
    private int currentLine;
    private int firstLine;
    private char separator;
    private String[] parsedLine;

    public InputParser(String path, char separator, boolean skipHeader) throws IOException {
        this.data = Files.readAllLines(Paths.get(path));
        this.separator = separator;
        this.firstLine = skipHeader ? 1 : 0;
        this.currentLine = this.firstLine;
    }

    public boolean nextLine() {
        if (currentLine < data.size()) {
            parsedLine = data.get(currentLine++).split(String.valueOf(separator));
            return true;
        }
        return false;
    }

    public double[] getParameters() {
        int size = parsedLine.length - 1;
        double[] values = new double[size];
        for (int i = 0; i < size; ++i) {
            values[i] = Double.parseDouble(parsedLine[i]);
        }
        return values;
    }

    public int[] getOutput() {
        int[] result = new int[1];
        result[0] = Integer.parseInt(parsedLine[parsedLine.length - 1]);
        return result;
    }

    public void begin() {
        currentLine = firstLine;
    }
}
