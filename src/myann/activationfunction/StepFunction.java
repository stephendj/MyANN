package myann.activationfunction;

public class StepFunction extends ActivationFunction {

    @Override
    public double calculateOutput(double net) {
        return Double.compare(net, 0) >= 0 ? 1 : 0;
    }

}
