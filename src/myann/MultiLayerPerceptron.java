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
        for (int i = 0; i < instances.numInstances(); ++i) {
            forwardChaining(instances.instance(i));
            backwardPropagation(instances.instance(i));
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
                        input.add(m_HiddenLayer.get(i - 1).get(k).getOutput());
                    }
                    m_HiddenLayer.get(i).get(j).calculateOutput(input);
                }
            }
        }

        //output layer
        for (int j = 0; j < m_OutputLayer.size(); ++j) {
            List<Double> input = new ArrayList<>();
            for (int k = 0; k < m_HiddenLayer.get(m_HiddenLayer.size() - 1).size(); ++k) { //layer sebelumnya
                input.add(m_HiddenLayer.get(m_HiddenLayer.size() - 1).get(k).getOutput());
            }
            m_OutputLayer.get(j).calculateOutput(input);

            System.out.println(m_OutputLayer.get(j).getOutput());
        }
    }

    public void printNewBiasWeight(double newBiasWeight) {
        System.out.println("newBiasWeight: " + newBiasWeight);
    }

    public void printNewWeights(List<Double> newWeights) {
        for (int i = 0; i < newWeights.size(); ++i) {
            System.out.println("newWeights " + i + ": " + newWeights.get(i));
        }
    }

    public void printErrorNow(List<Double> errorNow) {
        for (int i = 0; i < errorNow.size(); ++i) {
            System.out.println("errorNow " + i + ": " + errorNow.get(i));
        }
    }

    public void backwardPropagation(Instance instance) {
        List<Double> errorNext = new ArrayList<>();
        List<Double> errorNow = new ArrayList<>();

        // calculate error
        if (instance.classAttribute().isNumeric()) {
            double error = m_OutputLayer.get(0).getOutput() * (1 - m_OutputLayer.get(0).getOutput()) * (instance.classValue() - m_OutputLayer.get(0).getOutput());
            errorNow.add(error);
        } else if (instance.classIndex() == 2) { // nominal binary
            double error = m_OutputLayer.get(0).getOutput() * (1 - m_OutputLayer.get(0).getOutput()) * (instance.classIndex() - m_OutputLayer.get(0).getOutput());
            errorNow.add(error);
        } else { // nominal multiclass
            double error;
            for (int i = 0; i < m_OutputLayer.size(); ++i) {
                if (i == instance.classIndex()) {
                    error = m_OutputLayer.get(i).getOutput() * (1 - m_OutputLayer.get(i).getOutput()) * (1 - m_OutputLayer.get(i).getOutput());
                } else {
                    error = m_OutputLayer.get(i).getOutput() * (1 - m_OutputLayer.get(i).getOutput()) * (0 - m_OutputLayer.get(i).getOutput());
                }
                errorNow.add(error);
            }
        }

        // print errorNow
        printErrorNow(errorNow);

        // update weight outputLayer
        for (int j = 0; j < m_OutputLayer.size(); j++) {
            List<Double> newWeights = new ArrayList<>();
            for (int k = 0; k < m_HiddenLayer.get(m_HiddenLayer.size() - 1).size(); k++) {
                double newWeight = m_OutputLayer.get(j).getWeight().get(k) + m_LearningRate * errorNow.get(j) * m_HiddenLayer.get(m_HiddenLayer.size() - 1).get(k).getOutput();
                newWeights.add(newWeight);
            }
            m_OutputLayer.get(j).setWeight(newWeights);
            
            // update biasWeight
            double newBiasWeight = m_OutputLayer.get(j).getBiasWeight() + m_LearningRate * errorNow.get(j) * m_OutputLayer.get(j).getBias();
            m_OutputLayer.get(j).setBiasWeight(newBiasWeight);
            
            // print newWeights
            printNewWeights(newWeights);
            printNewBiasWeight(newBiasWeight);
        }
        
        for (int i = m_HiddenLayer.size() - 1; i >= 0; --i) {
            errorNext = new ArrayList<>(errorNow);
            errorNow = new ArrayList<>();

            if (m_HiddenLayer.size() == 1) { // instance output
                System.out.println("1");
                for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                    double error = 0.0;
                    for (int k = 0; k < m_OutputLayer.size(); k++) {
                        error += errorNext.get(k) * m_OutputLayer.get(k).getWeight().get(j);
                    }
                    double outputNeuron = m_HiddenLayer.get(i).get(j).getOutput();
                    error *= outputNeuron * (1 - outputNeuron);
                    errorNow.add(error);

                    //update weight
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                        double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k)
                            + m_LearningRate * error * instance.value(k);
                        newWeights.add(newWeight);
                    }
                    
                    // print newWeights
                    printNewWeights(newWeights);
                    
                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                }
                
            } else if (i == m_HiddenLayer.size() - 1) { // hidden output
                System.out.println("2");
                for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                    double error = 0.0;
                    for (int k = 0; k < m_OutputLayer.size(); k++) {
                        error += errorNext.get(k) * m_OutputLayer.get(k).getWeight().get(j);
                    }
                    double outputNeuron = m_HiddenLayer.get(i).get(j).getOutput();
                    error *= outputNeuron * (1 - outputNeuron);
                    errorNow.add(error);

                    //update weight
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                        double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k)
                            + m_LearningRate * error * m_HiddenLayer.get(i - 1).get(k).getOutput();
                        newWeights.add(newWeight);
                    }
                    
                    // print newWeights
                    printNewWeights(newWeights);

                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                }
            } else if (i == 0) { //instance hidden
                System.out.println("3");
                for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                    double error = 0.0;
                    for (int k = 0; k < m_HiddenLayer.get(i + 1).size(); k++) {
                        error += errorNext.get(k) * m_HiddenLayer.get(i + 1).get(k).getWeight().get(j);
                    }
                    double outputNeuron = m_HiddenLayer.get(i).get(j).getOutput();
                    error *= outputNeuron * (1 - outputNeuron);
                    errorNow.add(error);

                    //update weight
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                        double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k)
                            + m_LearningRate * error * instance.value(k);
                        newWeights.add(newWeight);
                    }
                    
                    // print newWeights
                    printNewWeights(newWeights);

                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                }
            } else { // hidden hidden
                System.out.println("4");
                for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                    double error = 0.0;
                    for (int k = 0; k < m_HiddenLayer.get(i + 1).size(); k++) {
                        error += errorNext.get(k) * m_HiddenLayer.get(i + 1).get(k).getWeight().get(j);
                    }
                    double outputNeuron = m_HiddenLayer.get(i).get(j).getOutput();
                    error *= outputNeuron * (1 - outputNeuron);
                    errorNow.add(error);

                    //update weight
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                        double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k)
                            + m_LearningRate * error * m_HiddenLayer.get(i - 1).get(k).getOutput();
                        newWeights.add(newWeight);
                    }
                    
                    // print newWeights
                    printNewWeights(newWeights);

                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                }
            }
            
            // update biasWeight
            for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                double newBiasWeight = m_HiddenLayer.get(i).get(j).getBiasWeight() + m_LearningRate * errorNow.get(j) * m_HiddenLayer.get(i).get(j).getBias();
                m_HiddenLayer.get(i).get(j).setBiasWeight(newBiasWeight);
                printNewBiasWeight(newBiasWeight);
            }
            
            // print errorNow
            printErrorNow(errorNow);

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
