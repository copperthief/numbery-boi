import org.nevec.rjm.BigDecimalMath;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.Bidi;

public class Node {
    public BigDecimal value;
    public BigDecimal z;

    public BigDecimal[] weights;
    public BigDecimal bias;

    //Todo: What if you trained an ANN to get worse by attempting to maximize the cost function? (positive instead of negative gradient

    public Node(int numInputs, double minWeight, double maxWeight, double minBias, double maxBias) {
        this.weights = randomWeights(numInputs, minWeight, maxWeight);
        this.bias = new BigDecimal(Math.random() * (maxBias - minBias) + minBias).setScale(32, BigDecimal.ROUND_HALF_EVEN);
    }

    public Node(BigDecimal value) {
        this.value = value;
    } //TODO adjust for bigdecimal

    private BigDecimal[] randomWeights (int numInputs, double min, double max) {
        BigDecimal[] weights = new BigDecimal[numInputs];
        for(int i = 0; i < numInputs; i++) {
            weights[i] = new BigDecimal(Math.random() * (max - min) + min).setScale(32, BigDecimal.ROUND_HALF_EVEN);
        }
        return weights;
    }

    public void calcValue (Node[] inputs) {
        BigDecimal value = new BigDecimal(0).setScale(32, BigDecimal.ROUND_HALF_EVEN);
        for(int i = 0; i < inputs.length; i++) {
            value = value.add(weights[i].multiply(inputs[i].value));
        }
        value = value.add(bias);
        this.z = value;
        this.value = sigmoid(value);
        //System.out.println("z: " + z + " a: " + sigmoid(z));
    }

    //private double sigmoid(double d) {
        //return 1 / (1 + Math.pow(Math.E, -d));
   // }

    private BigDecimal sigmoid (BigDecimal n) {
        return BigDecimal.ONE.setScale(32, RoundingMode.HALF_EVEN).divide(BigDecimal.ONE.add(BigDecimalMath.pow(BigDecimal.valueOf(Math.E).setScale(32, RoundingMode.HALF_EVEN), n.multiply(BigDecimal.valueOf(-1)))), RoundingMode.HALF_EVEN).setScale(32, RoundingMode.HALF_EVEN); //could the scaling of these cause problems??
    }
}
