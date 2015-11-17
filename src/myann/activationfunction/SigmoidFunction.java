package myann.activationfunction;

public class SigmoidFunction extends ActivationFunction {

    @Override
    public double calculateOutput(double net) {
        return (1 / (1 + Math.exp(-net)));
    }

}
