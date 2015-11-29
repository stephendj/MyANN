package myann.activationfunction;

public class SignFunction extends ActivationFunction {

    @Override
    public double calculateOutput(double net) {
        return Double.compare(net, 0) >= 0 ? 1 : -1;
    }
    
    @Override
    public String getName (){
        return  ActivationFunction.SIGN;
    }
}
