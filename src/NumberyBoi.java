import java.io.*;
import java.math.BigDecimal;

public class NumberyBoi {

    public static Network network = new Network(2, new int[]{784, 10});

    static File trainingData = new File("src//data//mnist_train.csv");
    static File testingData = new File("src//data//mnist_test.csv");

    static int[] trainingLabels = new int[60000];
    static int[][] trainingImages = new int[60000][784];
    static double[][]trainingImagesD = new double[60000][784];

    static int batchSize = 20;
    static double learningRate = 0.1;

    public static void main (String[] args) {
        network.init();


        try {
            BufferedReader trainingReader = new BufferedReader(new FileReader(trainingData));
            String line;
            int dataIndex = 0;
            while ((line = trainingReader.readLine()) != null) {
                trainingLabels[dataIndex] = Character.getNumericValue(line.charAt(0));
                String[] stringImageData = line.substring(2).split(",");

                for(int i = 0; i < stringImageData.length; i++) {
                    trainingImages[dataIndex][i] = Integer.parseInt(stringImageData[i]);
                }
                dataIndex++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < trainingImagesD.length; i++) {
            for (int j = 0; j < trainingImagesD[i].length; j++) {
                trainingImagesD[i][j] = trainingImages[i][j] / 255;
            }
        }

        /*
        network.setInput(trainingImages[0]);

        System.out.println(trainingLabels[0]);

        double[] results = network.classify();
        for (int i = 0; i < results.length; i++) System.out.println(i + " " + results[i]);

        System.out.println(network.cost(network.getOL(), trainingLabels[0]));

         */

        int exampleNum = 0;
        while (exampleNum < 60000) {
            network.feedForward(trainingImagesD[exampleNum]);
            BigDecimal cost = network.cost(network.getOL(), trainingLabels[exampleNum]);
            network.backprop(trainingLabels[exampleNum]);

           // System.out.println("Expected: " + trainingLabels[exampleNum]);
            for (int j = 0; j < network.getOL().length; j++) {
                //System.out.println(j + ": " + network.getOL()[j].value);
            }
           // System.out.println("Cost: " + cost);
            System.out.println(exampleNum + ": " + cost);

            exampleNum++;
        }

        /* stochastic stuff
        int examplesCompleted = 0;
        while (true) {
            for(int i = examplesCompleted; i < examplesCompleted + batchSize; i++) {
                //feedforward, cost, backprop
                network.cost(network.feedForward(), trainingLabels[i]);
            }
            //gradient descent
            examplesCompleted += batchSize;
        }

         */
    }

}
