package myann;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {
    
    private List<Double> m_Input = new ArrayList<>();
    private List<Double> m_Weight = new ArrayList<>();
    private ActivationFunction activationFunction;

    public Neuron(String afType) {
        m_Input.add(1.0);
        Random random = new Random();
        m_Weight.add(random.nextDouble());
        if(afType.equalsIgnoreCase("sign")) {
            activationFunction = new SignFunction();
        } else if(afType.equalsIgnoreCase("step")) {
            activationFunction = new StepFunction();
        } else if(afType.equalsIgnoreCase("sigmoid")) {
            activationFunction = new SigmoidFunction();
        }
    }
    
    public Neuron(double biasWeight, String afType) {
        m_Input.add(1.0);
        m_Weight.add(biasWeight);
        if(afType.equalsIgnoreCase("sign")) {
            activationFunction = new SignFunction();
        } else if(afType.equalsIgnoreCase("step")) {
            activationFunction = new StepFunction();
        } else if(afType.equalsIgnoreCase("sigmoid")) {
            activationFunction = new SigmoidFunction();
        }
    }
    
    public List<Double> getInput() {
        return m_Input;
    }

    public void setInput(List<Double> m_Input) {
        this.m_Input = m_Input;
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
    
    public void addInput(double input, double weight) {
        m_Input.add(input);
        m_Weight.add(weight);
    }
    
    public double calculateNetFunction() {
        double sum = 0;
        for(int i = 0; i < m_Input.size(); ++i) {
            sum += m_Input.get(i) * m_Weight.get(i);
        }
        return sum;
    }
}
