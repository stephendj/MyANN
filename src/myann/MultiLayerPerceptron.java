package myann;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class MultiLayerPerceptron extends Classifier {

    private List<List<Neuron>> m_HiddenLayer = new ArrayList<>();
    private List<Neuron> m_OutputLayer = new ArrayList<>();
    private int m_MaxIteration;
    private double m_LearningRate;
    private double m_Momentum;
    private Instances m_Instances;

    public MultiLayerPerceptron(List<List<Neuron>> hiddenLayer, List<Neuron> outputLayer, int m_MaxIteration, double m_LearningRate, double m_Momentum, Instances m_Instances) {
        this.m_HiddenLayer = hiddenLayer;
        this.m_OutputLayer = outputLayer;
        this.m_MaxIteration = m_MaxIteration;
        this.m_LearningRate = m_LearningRate;
        this.m_Momentum = m_Momentum;
        this.m_Instances = m_Instances;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        for (int i =0; i <instances.numInstances(); ++i) {
            forwardChaining(instances.instance(i));
        }
    }

    public void forwardChaining(Instance instance) {
        for (int i = 0; i < m_HiddenLayer.size(); ++i) {
            if (i == 0) { //dari instance
                for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                    m_HiddenLayer.get(i).get(j).calculateOutput(instance);
                }
            } else {
                for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                    List<Double> input = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i - 1).size(); ++k) { //layer sebelumnya
                        input.add(m_HiddenLayer.get(i-1).get(k).getOutput());
                    }
                    m_HiddenLayer.get(i).get(j).calculateOutput(input);
                }
            }
        }
        
        //output layer
        for (int j = 0; j < m_OutputLayer.size(); ++j) {
            List<Double> input = new ArrayList<>();
            for (int k = 0; k < m_HiddenLayer.get(m_HiddenLayer.size()-1).size(); ++k) { //layer sebelumnya
                input.add(m_HiddenLayer.get(m_HiddenLayer.size()-1).get(k).getOutput());
            }
            m_OutputLayer.get(j).calculateOutput(input);
            
            System.out.println(m_OutputLayer.get(j).getOutput());
        }
    }
    
    public void backwardPropagation (Instance instance) {
        List<Double> errorNext =new ArrayList<>();
        List<Double> errorNow =new ArrayList<>();
        
        for (int j = 0; j < m_OutputLayer.size(); j++) {
            for (int k = 0; k < m_HiddenLayer.get(m_HiddenLayer.size()-1).size(); k++) {
                //code not implemented
            }
        }
        
        for (int i =  m_HiddenLayer.size()-1; i >= 0; --i) {
            errorNext = new ArrayList<>(errorNow);
            errorNow = new ArrayList<>();
            
            if (m_HiddenLayer.size() == 1) { // 1 hidden layer
                for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                    double error = 0.0;
                    for (int k = 0; k < m_OutputLayer.size(); k++) {
                        error += errorNext.get(k) * m_OutputLayer.get(k).getWeight().get(j);
                    }
                    double outputNeuron= m_HiddenLayer.get(i).get(j).getOutput();
                    error *= outputNeuron * (1-outputNeuron);
                    errorNow.add(error);
                    
                    //update weight
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                       double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k) + 
                               m_LearningRate * error * instance.value(k);
                       newWeights.add(newWeight);
                    }
                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                }
            } else if (i == m_HiddenLayer.size()-1) { // hidden output
                for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                    double error = 0.0;
                    for (int k = 0; k < m_OutputLayer.size(); k++) {
                        error += errorNext.get(k) * m_OutputLayer.get(k).getWeight().get(j);
                    }
                    double outputNeuron= m_HiddenLayer.get(i).get(j).getOutput();
                    error *= outputNeuron * (1-outputNeuron);
                    errorNow.add(error);

                    //update weight
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                       double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k) + 
                               m_LearningRate * error * m_HiddenLayer.get(i-1).get(k).getOutput();
                       newWeights.add(newWeight);
                    }
                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                } 
            } else if (i == 0) { //instance hidden
                for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                    double error = 0.0;
                    for (int k = 0; k < m_HiddenLayer.get(i+1).size(); k++) {
                        error += errorNext.get(k) * m_HiddenLayer.get(i+1).get(k).getWeight().get(j);
                    }
                    double outputNeuron= m_HiddenLayer.get(i).get(j).getOutput();
                    error *= outputNeuron * (1-outputNeuron);
                    errorNow.add(error);

                    //update weight
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                       double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k) + 
                               m_LearningRate * error * instance.value(k);
                       newWeights.add(newWeight);
                    }
                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                } 
            } else { // hidden hidden
                for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                    double error = 0.0;
                    for (int k = 0; k < m_HiddenLayer.get(i+1).size(); k++) {
                        error += errorNext.get(k) * m_HiddenLayer.get(i+1).get(k).getWeight().get(j);
                    }
                    double outputNeuron= m_HiddenLayer.get(i).get(j).getOutput();
                    error *= outputNeuron * (1-outputNeuron);
                    errorNow.add(error);

                    //update weight
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                       double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k) + 
                               m_LearningRate * error * m_HiddenLayer.get(i-1).get(k).getOutput();
                       newWeights.add(newWeight);
                    }
                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                }
            }
        }
    }
    
    public List<List<Neuron>> getHiddenLayer() {
        return m_HiddenLayer;
    }

    public void setHiddenLayer(List<List<Neuron>> hiddenLayer) {
        this.m_HiddenLayer = hiddenLayer;
    }

    public List<Neuron> getOutputLayer() {
        return m_OutputLayer;
    }

    public void setOutputLayer(List<Neuron> outputLayer) {
        this.m_OutputLayer = outputLayer;
    }

    public int getM_MaxIteration() {
        return m_MaxIteration;
    }

    public void setM_MaxIteration(int m_MaxIteration) {
        this.m_MaxIteration = m_MaxIteration;
    }

    public double getM_LearningRate() {
        return m_LearningRate;
    }

    public void setM_LearningRate(double m_LearningRate) {
        this.m_LearningRate = m_LearningRate;
    }

    public double getM_Momentum() {
        return m_Momentum;
    }

    public void setM_Momentum(double m_Momentum) {
        this.m_Momentum = m_Momentum;
    }

    public Instances getM_Instances() {
        return m_Instances;
    }

    public void setM_Instances(Instances m_Instances) {
        this.m_Instances = m_Instances;
    }

}
