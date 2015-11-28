package myann;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public abstract class SingleLayerPerceptron extends Classifier {

    private int m_MaxIteration;
    private Instances m_Instances;
    private List<Neuron> m_Neuron;
    private double m_LearningRate;
    private double m_Momentum;

    /**
     *
     * @param m_MaxIteration maximum iteration
     * @param m_Neuron neuron
     * @param m_LearningRate learning rate
     * @param m_Momentum momentum
     */
    public SingleLayerPerceptron(int m_MaxIteration, List<Neuron> m_Neuron,
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
    public List<Double> calculateOutput(Instance instance) {
        List<Double> outputs = new ArrayList<>();
        
        for(Neuron neuron : m_Neuron) {
            neuron.calculateOutput(instance);
            outputs.add(neuron.getOutput());
        }
        
        return outputs;
    }
    
    /**
     *
     * @param idx the index of the instance
     * @return delta bias weight
     */
    public List<Double> calculateDeltaBiasWeight(int idx) {
        List<Double> deltaBiasWeights = new ArrayList<>();
        // calculate error
        if (m_Instances.classAttribute().isNumeric() || m_Instances.classAttribute().numValues() == 2) {
            double error = m_Instances.instance(idx).classValue() - calculateOutput(m_Instances.instance(idx)).get(0);
            deltaBiasWeights.add(m_LearningRate * m_Neuron.get(0).getBias() * error);
        } else { // nominal multiclass
            double error;
            for (int i = 0; i < m_Neuron.size(); ++i) {
                if (i == m_Instances.instance(idx).classIndex()) {
                    error = 1 - m_Neuron.get(i).getOutput();
                } else {
                    error = 0 - m_Neuron.get(i).getOutput();
                }
                deltaBiasWeights.add(m_LearningRate * m_Neuron.get(i).getBias() * error);
            }
        }
        
        return deltaBiasWeights;
    }
    
    /**
     *
     * @param idx the index of the instance
     * @return list of delta weight
     */
    public List<List<Double>> calculateDeltaWeight(int idx) {
        List<List<Double>> deltaWeights = new ArrayList<>();
        
        if (m_Instances.classAttribute().isNumeric() || m_Instances.classAttribute().numValues() == 2) {
            double error = m_Instances.instance(idx).classValue() - calculateOutput(m_Instances.instance(idx)).get(0);
            List<Double> deltaWeightsPerNeuron = new ArrayList<>();
            for (int i = 0; i < m_Neuron.get(0).getWeight().size(); ++i) {
                double input = m_Instances.instance(idx).value(i);
                deltaWeightsPerNeuron.add(m_LearningRate * input * error);
            }
            deltaWeights.add(deltaWeightsPerNeuron);
        } else { // nominal multiclass
            double error;
            for (int i = 0; i < m_Neuron.size(); ++i) {
                if (i == m_Instances.instance(idx).classIndex()) {
                    error = 1 - m_Neuron.get(i).getOutput();
                } else {
                    error = 0 - m_Neuron.get(i).getOutput();
                }
                List<Double> deltaWeightsPerNeuron = new ArrayList<>();
                for (int j = 0; j < m_Neuron.get(i).getWeight().size(); ++j) {
                    double input = m_Instances.instance(idx).value(j);
                    deltaWeightsPerNeuron.add(m_LearningRate * input * error);
                }
                deltaWeights.add(deltaWeightsPerNeuron);
            }
        }
        
        return deltaWeights;
    }

    /**
     *
     * @param deltaWeights list of the delta weight
     * @param deltaBiasWeight delta of bias weight
     */
    public void updateWeights(List<List<Double>> deltaWeights, List<Double> deltaBiasWeight) {
        for(int i = 0; i < m_Neuron.size(); ++i) {
            List<Double> newWeight = new ArrayList<>();
            for (int j = 0; j < m_Neuron.get(i).getWeight().size(); ++j) {
                newWeight.add(m_Neuron.get(i).getWeight().get(j) + deltaWeights.get(i).get(j));
            }
            m_Neuron.get(i).setWeight(newWeight);
            m_Neuron.get(i).setBiasWeight(m_Neuron.get(i).getBiasWeight() + deltaBiasWeight.get(i));
        }
    }
    /**
     *
     * @return mean square error for 1 epoch
     */
    public double calculateMSE() {
        double sum = 0;
        
        int maxOutputIndex = 0;
        for (int i = 1; i < m_Neuron.size(); ++i) {
            if (m_Neuron.get(i).getOutput() > m_Neuron.get(maxOutputIndex).getOutput()) {
                maxOutputIndex = i;
            }
        }
        for (int i = 0; i < m_Instances.numInstances(); ++i) {
            
            double error = 0;
            
            if (m_Instances.classAttribute().isNumeric() || m_Instances.classAttribute().numValues() == 2) {
                error = m_Instances.instance(i).classValue() - calculateOutput(m_Instances.instance(i)).get(maxOutputIndex);
            } else { // nominal multiclass
                for (int j = 0; j < m_Neuron.size(); ++j) {
                    if (j == m_Instances.instance(i).classIndex()) {
                        error = 1 - calculateOutput(m_Instances.instance(i)).get(maxOutputIndex);
                    } else {
                        error = 0 - calculateOutput(m_Instances.instance(i)).get(maxOutputIndex);
                    }
                }
            }
            
            sum += Math.pow(error, 2);
        }
        return 0.5 * sum;
    }

    /**
     * print the model of neuron
     */
    public void printModel() {
        for(int i = 0; i < m_Neuron.size(); ++i) {
            System.out.println("Neuron " + i);
            System.out.println("w_bias : " + m_Neuron.get(i).getBiasWeight());
            for (int j = 0; j < m_Neuron.get(i).getWeight().size(); ++j) {
                System.out.println("w" + j + " : " + m_Neuron.get(i).getWeight().get(j));
            }
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
    public List<Neuron> getNeuron() {
        return m_Neuron;
    }

    /**
     *
     * @param m_Neuron neuron
     */
    public void setNeuron(List<Neuron> m_Neuron) {
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
