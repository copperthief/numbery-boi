import org.nevec.rjm.BigDecimalMath;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class Network {
    Node[][] network;

    Network(int numLayers, int[] numNodes) {
        this.network = new Node[numLayers][];
        for(int i = 0; i < numLayers; i++) network[i] = new Node[numNodes[i]];
    }

    public void init() {
        //TODO: get input sorted
        for(int i = 1; i < network.length; i++) {
            initLayer(i);
        }
    }

    //TODO: throw error if input.length != network[0].length
    public void feedForward(double[] input) {
        for(int i = 0; i < network[0].length; i++) {
            network[0][i] = new Node(BigDecimal.valueOf(input[i]).setScale(32, RoundingMode.HALF_EVEN));
        }
        for(int i = 1; i < network.length; i++) {
            runLayer(i);
        }
        BigDecimal[] output = new BigDecimal[network[network.length - 1].length];
        for (int i = 0; i < output.length; i++) {
            output[i] = network[network.length - 1][i].value;
        }
    }

    public BigDecimal cost(Node[] OL, int expected) {
        BigDecimal cost = BigDecimal.ZERO.setScale(32, RoundingMode.HALF_EVEN);
        for(int i = 0; i < OL.length; i++) {
            if (i != expected) cost = cost.add(BigDecimalMath.pow(abs(OL[i].value).setScale(32, RoundingMode.HALF_EVEN), BigDecimal.valueOf(2)));
            else cost = cost.add(BigDecimalMath.pow(abs(OL[i].value.subtract(BigDecimal.ONE)).setScale(32, RoundingMode.HALF_EVEN), BigDecimal.valueOf(2))); //could cause scale problems
        }
        return cost.multiply(BigDecimal.valueOf(0.5));
    }

    //same as comment closest above
    public void initLayer(int layer) {
        for(int i = 0; i < network[layer].length; i++) {
            network[layer][i] = new Node(network[layer - 1].length, -4, 4, -3, 3);
        }
    }

    //throw error if layer number is 0 or doesn't exist
    public void runLayer(int layer) {
        for(int i = 0; i < network[layer].length; i++) {
            network[layer][i].calcValue(network[layer - 1]);
        }
    }

    public void backprop(int expected) {

        //what number does the network think it is?
        int calculated = 0;
        BigDecimal max = new BigDecimal(0).setScale(32, RoundingMode.HALF_EVEN);
        for (int j = 0; j < getOL().length; j++) {
            if (getOL()[j].value.compareTo(max) > 0) {
                max = getOL()[j].value;
                calculated = j;
            }
        }

        //System.out.println("E: " + expected + " C: " + calculated);

        int[] expectedArray = new int[getOL().length];
        for(int j = 0; j < expectedArray.length; j++) {
            if (j == expected) {
                expectedArray[j] = 1;
            } else {
                expectedArray[j] = 0;
            }
        }

        //NOTE: Because the input layer does not have error, error[n] corresponds to network[n+1]
        BigDecimal[][] error = new BigDecimal[network.length - 1][];
        for (int j = 0; j < error.length; j++) {
            error[j] = new BigDecimal[network[j + 1].length];
        }

        //System.out.println("calculating output error");

        //calc output error
        for (int i = 0; i < error[error.length - 1].length; i++) {
            error[error.length - 1][i] = getOL()[i].value.subtract(BigDecimal.valueOf(expectedArray[i])).multiply(sigPrime(getOL()[i].z)).setScale(32, RoundingMode.HALF_EVEN);
            //System.out.println("wearsttdyfukgikyfdhtsr");
            //System.out.println("Expected[i]: " + BigDecimal.valueOf(expectedArray[i]));
            //System.out.println("OL[i] - Expected[i]: " + getOL()[i].value.subtract(BigDecimal.valueOf(expectedArray[i])));
            //System.out.println("OL[i].z: " + getOL()[i].z);
            //System.out.println("sigprime: " + (sigPrime(getOL()[i].z)));
            //System.out.println(error[error.length - 1][i]);
            //System.out.println("wearsttdyfukgikyfdhtsr");

        }

        //System.out.println("backprop error");

        //backprop error
        for(int j = error.length - 2; j > 0; j--) {
            for(int k = 0; k < error[j].length; k++) {
                error[j][k] = BigDecimal.ZERO.setScale(32, RoundingMode.HALF_EVEN);
                for(int l = 0; l < error[j+1].length; l++) {
                    error[j][k] = error[j][k].add(network[j+2][l].weights[k].multiply(error[j+1][l]).multiply(sigPrime(network[j+1][k].z)));
                    //System.out.println("Q" + error[j][k]);
                }
            }
        }

        //System.out.println(" ewresgtdyfu");
            for (BigDecimal b : error[error.length - 1]) {
                //System.out.println(b);
            }

        //calc weight gradient
        BigDecimal[][][] deltaW = new BigDecimal[network.length - 1][][];
        for (int j = 0; j < deltaW.length; j++) {
            deltaW[j] = new BigDecimal[network[j + 1].length][];
            for (int k = 0; k < deltaW[j].length; k++) {
                deltaW[j][k] = new BigDecimal[network[j].length];
                for (int l = 0; l < deltaW[j][k].length; l++) {
                    //System.out.println(error[j][k]);
                    deltaW[j][k][l] = network[j][l].value.multiply(error[j][k]);
                }
            }
        }

        //update weights and biases
        for (int j = 0; j < deltaW.length; j++) {
            for (int k = 0; k < deltaW[j].length; k++) {
                for (int l = 0; l < deltaW[j][k].length; l++) {
                    network[j + 1][k].weights[l] = network[j + 1][k].weights[l].subtract(deltaW[j][k][l].multiply(BigDecimal.valueOf(NumberyBoi.learningRate)));
                }
            }
        }
        for(int j = 0; j < error.length; j++) {
            for(int k = 0; k < error[j].length; k++) {
                network[j + 1][k].bias = network[j + 1][k].bias.subtract(error[j][k].multiply(BigDecimal.valueOf(NumberyBoi.learningRate)));
            }
        }
    }

    public double sigmoid (double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    private BigDecimal sigmoid (BigDecimal n) {
        return BigDecimal.ONE.setScale(32, RoundingMode.HALF_EVEN).divide(BigDecimal.ONE.add(BigDecimalMath.pow(BigDecimal.valueOf(Math.E).setScale(32, RoundingMode.HALF_EVEN), n.multiply(BigDecimal.valueOf(-1)))), RoundingMode.HALF_EVEN).setScale(32, RoundingMode.HALF_EVEN); //could the scaling of these cause problems??
    }

    public double sigPrime (double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public BigDecimal sigPrime (BigDecimal x) {
        return sigmoid(x).multiply(BigDecimal.ONE.subtract(sigmoid(x)));
    }

    public BigDecimal abs (BigDecimal x) {
        return (x.compareTo(BigDecimal.valueOf(0)) < 0) ? x.multiply(BigDecimal.valueOf(-1)) : x;
    }

    //get input layer
    public Node[] getIL()  {
        return network[0];
    }

    public Node[] getHL(int n) {
        return network[n];
    }

    //get output layer
    public Node[] getOL() {
        return network[network.length - 1];
    }
}
