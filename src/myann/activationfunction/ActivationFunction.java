package myann.activationfunction;

import java.io.Serializable;

public abstract class ActivationFunction implements Serializable {

    public static final String NONE = "none";
    public static final String SIGN = "sign";
    public static final String SIGMOID = "sigmoid";
    public static final String STEP = "step";

    /**
     *
     * @param net value to be calculated
     * @return calculate the value based on the activation function
     */
    public abstract double calculateOutput(double net);
    public abstract String getName();

}
