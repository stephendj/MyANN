package myann;

public class StepFunction extends ActivationFunction {

    @Override
    public double calculateOutput(double net) {
        return net >= 0 ? 1 : 0;
    }

}
