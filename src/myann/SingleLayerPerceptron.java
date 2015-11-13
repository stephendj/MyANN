package myann;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class SingleLayerPerceptron extends Classifier {

    private int m_MaxIteration;
    private Instances m_Instances;
    private Neuron m_Neuron;
    private double m_LearningRate;
    private double m_Momentum;
    
    
    
    public List<Double> calculateDeltaWeight(int idx) {
        List<Double> deltaWeights = new ArrayList<>();
        double error = m_Instances.instance(idx).classValue() - m_Neuron.calculateOutput(m_Instances.instance(idx));
        for(int i = 0; i < m_Neuron.getWeight().size(); ++i) {
            double input = m_Instances.instance(idx).value(i+1);
            deltaWeights.add(m_LearningRate * input * error);
        }
        return deltaWeights;
    }
    
    public double calculateDeltaBiasWeight(int idx) {
        double error = m_Instances.instance(idx).classValue() - m_Neuron.calculateOutput(m_Instances.instance(idx));
        return m_LearningRate * m_Neuron.getBias() * error;
    }
    
    public double calculateMSE() {
        double sum =0;
        for(int i = 0; i < m_Instances.numInstances(); ++i) {
            double error = m_Instances.instance(i).classValue() - m_Neuron.calculateOutput(m_Instances.instance(i));
            sum += Math.pow(error, 2);
        }
        return 0.5 * sum;
    }
    
    public void printModel() {
        for(int i = 0; i < m_Neuron.getWeight().size(); ++i) {
            System.out.println(m_Neuron.getWeight().get(i));
        }
    }
}
