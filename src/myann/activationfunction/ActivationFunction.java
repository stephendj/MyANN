package myann.activationfunction;

public abstract class ActivationFunction {

    /**
     *
     * @param net value to be calculated
     * @return calculate the value based on the activation function
     */
    public abstract double calculateOutput(double net);
}
