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

    /**
     *
     * @param m_MaxIteration maximum iteration
     * @param m_Neuron neuron
     * @param m_LearningRate learning rate
     * @param m_Momentum momentum
     */
    public SingleLayerPerceptron(int m_MaxIteration, Neuron m_Neuron,
            double m_LearningRate, double m_Momentum) {
        this.m_MaxIteration = m_MaxIteration;
        this.m_Neuron = m_Neuron;
        this.m_LearningRate = m_LearningRate;
        this.m_Momentum = m_Momentum;
    }

    /**
     *
     * @param instance instance
     * @return output from the instance
     */
    public double calculateOutput(Instance instance) {
        m_Neuron.calculateOutput(instance);
        return m_Neuron.getOutput();
    }
    
    /**
     *
     * @param idx the index of the instance
     * @return delta bias weight
     */
    public double calculateDeltaBiasWeight(int idx) {
        double error = m_Instances.instance(idx).classValue() - calculateOutput(m_Instances.instance(idx));
        return m_LearningRate * m_Neuron.getBias() * error;
    }
    
    /**
     *
     * @param idx the index of the instance
     * @return list of delta weight
     */
    public List<Double> calculateDeltaWeight(int idx) {
        List<Double> deltaWeights = new ArrayList<>();
        double error = m_Instances.instance(idx).classValue() - calculateOutput(m_Instances.instance(idx));
        for (int i = 0; i < m_Neuron.getWeight().size(); ++i) {
            double input = m_Instances.instance(idx).value(i);
            deltaWeights.add(m_LearningRate * input * error);
        }
        
        return deltaWeights;
    }

    /**
     *
     * @param deltaWeights list of the delta weight
     * @param deltaBiasWeight delta of bias weight
     */
    public void updateWeights(List<Double> deltaWeights, double deltaBiasWeight) {
        List<Double> newWeight = new ArrayList<>();
        for (int i = 0; i < m_Neuron.getWeight().size(); ++i) {
            newWeight.add(m_Neuron.getWeight().get(i) + deltaWeights.get(i));
        }
        m_Neuron.setWeight(newWeight);
        m_Neuron.setBiasWeight(m_Neuron.getBiasWeight() + deltaBiasWeight);
    }
    /**
     *
     * @return mean square error for 1 epoch
     */
    public double calculateMSE() {
        double sum = 0;
        for (int i = 0; i < m_Instances.numInstances(); ++i) {
            double error = m_Instances.instance(i).classValue() - calculateOutput(m_Instances.instance(i));
            sum += Math.pow(error, 2);
        }
        return 0.5 * sum;
    }

    /**
     * print the model of neuron
     */
    public void printModel() {
        System.out.println("w_bias : " + m_Neuron.getBiasWeight());
        for (int i = 0; i < m_Neuron.getWeight().size(); ++i) {
            System.out.println("w" + i + " : " + m_Neuron.getWeight().get(i));
        }
    }

    /**
     *
     * @return maximum iteration
     */
    public int getMaxIteration() {
        return m_MaxIteration;
    }

    /**
     *
     * @param m_MaxIteration maximum iteration
     */
    public void setMaxIteration(int m_MaxIteration) {
        this.m_MaxIteration = m_MaxIteration;
    }

    /**
     *
     * @return instances
     */
    public Instances getInstances() {
        return m_Instances;
    }

    /**
     *
     * @param m_Instances data train
     */
    public void setInstances(Instances m_Instances) {
        this.m_Instances = m_Instances;
    }

    /**
     *
     * @return neuron
     */
    public Neuron getNeuron() {
        return m_Neuron;
    }

    /**
     *
     * @param m_Neuron neuron
     */
    public void setNeuron(Neuron m_Neuron) {
        this.m_Neuron = m_Neuron;
    }

    /**
     *
     * @return learning rate
     */
    public double getLearningRate() {
        return m_LearningRate;
    }

    /**
     *
     * @param m_LearningRate learning rate
     */
    public void setLearningRate(double m_LearningRate) {
        this.m_LearningRate = m_LearningRate;
    }

    /**
     *
     * @return momentum
     */
    public double getMomentum() {
        return m_Momentum;
    }

    /**
     *
     * @param m_Momentum momentum
     */
    public void setMomentum(double m_Momentum) {
        this.m_Momentum = m_Momentum;
    }

}
