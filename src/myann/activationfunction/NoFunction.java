package myann.activationfunction;

public class NoFunction extends ActivationFunction {

    @Override
    public double calculateOutput(double net) {
        return net;
    }
    
    @Override
    public String getName (){
        return  ActivationFunction.NONE;
    }
}
