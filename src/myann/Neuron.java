package myann;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.core.Instance;

public class Neuron {
    
    private List<Double> m_Weight = new ArrayList<>();
    private double m_Bias = 1;
    private double m_BiasWeight;
    private ActivationFunction activationFunction;

    public Neuron(String afType, List<Double> weight) {
        Random random = new Random();
        m_BiasWeight = random.nextDouble();
        m_Weight.addAll(weight);
        if(afType.equalsIgnoreCase("sign")) {
            activationFunction = new SignFunction();
        } else if(afType.equalsIgnoreCase("step")) {
            activationFunction = new StepFunction();
        } else if(afType.equalsIgnoreCase("sigmoid")) {
            activationFunction = new SigmoidFunction();
        }
    }
    
    public Neuron(double biasWeight, String afType, List<Double> weight) {
        m_BiasWeight = biasWeight;
        m_Weight.addAll(weight);
        if(afType.equalsIgnoreCase("sign")) {
            activationFunction = new SignFunction();
        } else if(afType.equalsIgnoreCase("step")) {
            activationFunction = new StepFunction();
        } else if(afType.equalsIgnoreCase("sigmoid")) {
            activationFunction = new SigmoidFunction();
        }
    }
    
    private double calculateNetFunction(Instance instance) {
        double sum = 0;
        for(int i = 0; i < instance.numAttributes(); ++i) {
            sum += instance.value(i+1) * m_Weight.get(i);
        }
        sum += m_Bias * m_BiasWeight;
        return sum;
    }
    
    public double calculateOutput(Instance instance) {
        return activationFunction.calculateOutput(calculateNetFunction(instance));
    }

    public List<Double> getWeight() {
        return m_Weight;
    }

    public void setWeight(List<Double> m_Weight) {
        this.m_Weight = m_Weight;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public double getBias() {
        return m_Bias;
    }
    
    public void setBias(double m_Bias) {
        this.m_Bias = m_Bias;
    }
    
    public double getBiasWeight() {
        return m_BiasWeight;
    }
    
    public void setBiasWeight(double m_BiasWeight) {
        this.m_BiasWeight = m_BiasWeight;
    }
}
