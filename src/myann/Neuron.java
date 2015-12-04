package myann;

import java.io.Serializable;
import myann.activationfunction.ActivationFunction;
import myann.activationfunction.SignFunction;
import myann.activationfunction.StepFunction;
import myann.activationfunction.SigmoidFunction;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import myann.activationfunction.NoFunction;
import weka.core.Instance;

public class Neuron implements Serializable {

    private List<Double> m_Weight = new ArrayList<>();
    private double m_Bias = 1;
    private double m_BiasWeight;
    private ActivationFunction activationFunction;
    private double m_Output;
    
    /**
     *
     * @param afType activation type
     * @param numInput the number of weight to be randomized
     */
    public Neuron(String afType, int numInput) {
        Random random = new Random();
        m_BiasWeight = random.nextDouble();
        for(int i = 0; i < numInput; ++i) {
            m_Weight.add(random.nextDouble());
        }
        if (afType.equalsIgnoreCase("sign")) {
            activationFunction = new SignFunction();
        } else if (afType.equalsIgnoreCase("step")) {
            activationFunction = new StepFunction();
        } else if (afType.equalsIgnoreCase("sigmoid")) {
            activationFunction = new SigmoidFunction();
        } else if (afType.equalsIgnoreCase("no")) {
            activationFunction = new NoFunction();
        } 
    }

    /**
     *
     * @param afType activation type
     * @param biasWeight bias weight
     * @param weight list of neuron weights 
     */
    public Neuron(String afType, double biasWeight, List<Double> weight) {
        m_BiasWeight = biasWeight;
        m_Weight.addAll(weight);
        if (afType.equalsIgnoreCase("sign")) {
            activationFunction = new SignFunction();
        } else if (afType.equalsIgnoreCase("step")) {
            activationFunction = new StepFunction();
        } else if (afType.equalsIgnoreCase("sigmoid")) {
            activationFunction = new SigmoidFunction();
        } else if (afType.equalsIgnoreCase("none")) {
            activationFunction = new NoFunction();
        } 
    }

    private double calculateNetFunction(Instance instance) {
        double sum = 0;
        for (int i = 0; i < instance.numAttributes() - 1; ++i) {
            sum += instance.value(i) * m_Weight.get(i);
        }
        sum += m_Bias * m_BiasWeight;
        return sum;
    }
    
    private double calculateNetFunction(List<Double> input) {
        double sum = 0;
        for (int i = 0; i < input.size() ; ++i) {
            sum += input.get(i) * m_Weight.get(i);
        }
        sum += m_Bias * m_BiasWeight;
        return sum;
    }

    /**
     *
     * @param instance the instance
     * @return output from activation function
     */
    public void calculateOutput(Instance instance) {
        m_Output = activationFunction.calculateOutput(calculateNetFunction(instance));
    }
    
    /**
     *
     * @param input the input
     * @return output from activation function
     */
    public void calculateOutput(List<Double> input ) {
        m_Output = activationFunction.calculateOutput(calculateNetFunction(input));
    }

    /**
     * print the model of neuron
     */
    public void printModel() {
        System.out.println("w_bias : " + m_BiasWeight);
        for (int i = 0; i < m_Weight.size(); ++i) {
            System.out.println("w" + i + " : " + m_Weight.get(i));
        }
    }

    /**
     *
     * @return list of weight
     */
    public List<Double> getWeight() {
        return m_Weight;
    }

    /**
     *
     * @param m_Weight list of weight
     */
    public void setWeight(List<Double> m_Weight) {
        this.m_Weight = m_Weight;
    }

    /**
     *
     * @return activation function
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    /**
     *
     * @param activationFunction activation function
     */
    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     *
     * @return bias value
     */
    public double getBias() {
        return m_Bias;
    }

    /**
     *
     * @param m_Bias bias value
     */
    public void setBias(double m_Bias) {
        this.m_Bias = m_Bias;
    }

    /**
     *
     * @return bias weight
     */
    public double getBiasWeight() {
        return m_BiasWeight;
    }

    /**
     *
     * @param m_BiasWeight bias weight
     */
    public void setBiasWeight(double m_BiasWeight) {
        this.m_BiasWeight = m_BiasWeight;
    }
    
    /**
     *
     * @return output
     */
    public double getOutput() {
        return m_Output;
    }
}
