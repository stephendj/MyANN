package myann;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public abstract class SingleLayerPerceptron extends Classifier {

    private int m_MaxIteration;
    private Instances m_Instances;
    private Neuron m_Neuron;
    private double m_LearningRate;
    private double m_Momentum;

    public SingleLayerPerceptron(int m_MaxIteration, Neuron m_Neuron,
            double m_LearningRate, double m_Momentum) {
        this.m_MaxIteration = m_MaxIteration;
        this.m_Neuron = m_Neuron;
        this.m_LearningRate = m_LearningRate;
        this.m_Momentum = m_Momentum;
    }

    public List<Double> calculateDeltaWeight(int idx) {
        List<Double> deltaWeights = new ArrayList<>();
        double error = m_Instances.instance(idx).classValue() - m_Neuron.calculateOutput(m_Instances.instance(idx));
        for (int i = 0; i < m_Neuron.getWeight().size(); ++i) {
            double input = m_Instances.instance(idx).value(i);
            deltaWeights.add(m_LearningRate * input * error);
        }
        
        return deltaWeights;
    }

    public double calculateDeltaBiasWeight(int idx) {
        double error = m_Instances.instance(idx).classValue() - m_Neuron.calculateOutput(m_Instances.instance(idx));
        return m_LearningRate * m_Neuron.getBias() * error;
    }

    public double calculateMSE() {
        double sum = 0;
        for (int i = 0; i < m_Instances.numInstances(); ++i) {
            double error = m_Instances.instance(i).classValue() - m_Neuron.calculateOutput(m_Instances.instance(i));
            sum += Math.pow(error, 2);
        }
        return 0.5 * sum;
    }

    public double calculateOutput(Instance instance) {
        return m_Neuron.calculateOutput(instance);
    }

    public void updateWeights(List<Double> deltaWeights, double deltaBiasWeight) {
        List<Double> newWeight = new ArrayList<>();
        for (int i = 0; i < m_Neuron.getWeight().size(); ++i) {
            newWeight.add(m_Neuron.getWeight().get(i) + deltaWeights.get(i));
        }
        m_Neuron.setWeight(newWeight);
        m_Neuron.setBiasWeight(m_Neuron.getBiasWeight() + deltaBiasWeight);
    }

    public void printModel() {
        System.out.println("w_bias : " + m_Neuron.getBiasWeight());
        for (int i = 0; i < m_Neuron.getWeight().size(); ++i) {
            System.out.println("w" + i + " : " + m_Neuron.getWeight().get(i));
        }
    }

    public int getMaxIteration() {
        return m_MaxIteration;
    }

    public void setMaxIteration(int m_MaxIteration) {
        this.m_MaxIteration = m_MaxIteration;
    }

    public Instances getInstances() {
        return m_Instances;
    }

    public void setInstances(Instances m_Instances) {
        this.m_Instances = m_Instances;
    }

    public Neuron getNeuron() {
        return m_Neuron;
    }

    public void setNeuron(Neuron m_Neuron) {
        this.m_Neuron = m_Neuron;
    }

    public double getLearningRate() {
        return m_LearningRate;
    }

    public void setLearningRate(double m_LearningRate) {
        this.m_LearningRate = m_LearningRate;
    }

    public double getMomentum() {
        return m_Momentum;
    }

    public void setMomentum(double m_Momentum) {
        this.m_Momentum = m_Momentum;
    }

}
