package myann;

public class SignFunction extends ActivationFunction {

    @Override
    public double calculateOutput(double net) {
        return net >= 0 ? 1 : -1;
    }
    
}
