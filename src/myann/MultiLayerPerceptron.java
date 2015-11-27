package myann;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class MultiLayerPerceptron extends Classifier {

    private List<List<Neuron>> m_HiddenLayer = new ArrayList<>();
    private List<Neuron> m_OutputLayer = new ArrayList<>();
    private Instances m_Instances;
    private int m_MaxIteration;
    private double m_LearningRate;
    private double m_Momentum;
    private double m_Threshold;

    public MultiLayerPerceptron(List<List<Neuron>> hiddenLayer,
        List<Neuron> outputLayer, Instances m_Instances, int m_MaxIteration, double m_LearningRate, double m_Momentum, double m_Threshold) {
        this.m_HiddenLayer = hiddenLayer;
        this.m_OutputLayer = outputLayer;
        this.m_Instances = m_Instances;
        this.m_MaxIteration = m_MaxIteration;
        this.m_LearningRate = m_LearningRate;
        this.m_Momentum = m_Momentum;
        this.m_Threshold = m_Threshold;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        double mse = m_Threshold;
        int epoch = 1;
        while (m_MaxIteration == 0 && mse < m_Threshold || epoch <= m_MaxIteration) {
            System.out.println("Epoch " + epoch);

            for (int i = 0; i < instances.numInstances(); ++i) {
                forwardChaining(instances.instance(i));
                backwardPropagation(instances.instance(i));
            }

            mse = calculateMSE();
            System.out.println("mse: " + mse);

            ++epoch;
        }
    }

    public void forwardChaining(Instance instance) {
        for (int i = 0; i < m_HiddenLayer.size(); ++i) {
            if (i == 0) { // input from instance
                for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                    m_HiddenLayer.get(i).get(j).calculateOutput(instance);
                }
            } else {
                for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                    List<Double> input = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i - 1).size(); ++k) { // previous hidden layer
                        input.add(m_HiddenLayer.get(i - 1).get(k).getOutput());
                    }
                    m_HiddenLayer.get(i).get(j).calculateOutput(input);
                }
            }
        }

        //output layer
        for (int j = 0; j < m_OutputLayer.size(); ++j) {
            List<Double> input = new ArrayList<>();
            for (int k = 0; k < m_HiddenLayer.get(m_HiddenLayer.size() - 1).size(); ++k) { // last hidden layer
                input.add(m_HiddenLayer.get(m_HiddenLayer.size() - 1).get(k).getOutput());
            }
            m_OutputLayer.get(j).calculateOutput(input);

            System.out.println("output: " + m_OutputLayer.get(j).getOutput());
        }
    }

    public void backwardPropagation(Instance instance) {
        List<Double> errorNext = new ArrayList<>();
        List<Double> errorNow = new ArrayList<>();

        // calculate error
        if (instance.classAttribute().isNumeric()) {
            double error = m_OutputLayer.get(0).getOutput() * (1 - m_OutputLayer.get(0).getOutput()) * (instance.classValue() - m_OutputLayer.get(0).getOutput());
            errorNow.add(error);
        } else if (instance.classAttribute().numValues() == 2) { // nominal binary
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

    public double calculateMSE() {
        // search for maximum output index
        int maxOutputIndex = 0;
        for (int i = 1; i < m_OutputLayer.size(); ++i) {
            if (m_OutputLayer.get(i).getOutput() > m_OutputLayer.get(maxOutputIndex).getOutput()) {
                maxOutputIndex = i;
            }
        }

        double mse = 0;
        for (int n = 0; n < m_Instances.numInstances(); ++n) {
            // forward chaining hidden layer
            for (int i = 0; i < m_HiddenLayer.size(); ++i) {
                if (i == 0) { // input from instance
                    for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                        m_HiddenLayer.get(i).get(j).calculateOutput(m_Instances.instance(n));
                    }
                } else { // input from previous hidden layer
                    for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                        List<Double> inputs = new ArrayList<>();
                        for (int k = 0; k < m_HiddenLayer.get(i - 1).size(); ++k) { // previous hidden layer
                            inputs.add(m_HiddenLayer.get(i - 1).get(k).getOutput());
                        }
                        m_HiddenLayer.get(i).get(j).calculateOutput(inputs);
                    }
                }
            }

            // forward chaining maximum output layer
            List<Double> input = new ArrayList<>();
            for (int k = 0; k < m_HiddenLayer.get(m_HiddenLayer.size() - 1).size(); ++k) { // last hidden layer
                input.add(m_HiddenLayer.get(m_HiddenLayer.size() - 1).get(k).getOutput());
            }
            m_OutputLayer.get(maxOutputIndex).calculateOutput(input);

            double error = 0;
            if (m_Instances.classAttribute().isNumeric() || m_Instances.classAttribute().numValues() == 2) {
                error = m_Instances.instance(n).classValue() - m_OutputLayer.get(maxOutputIndex).getOutput();
            } else { // nominal multiclass
                for (int i = 0; i < m_OutputLayer.size(); ++i) {
                    if (i == m_Instances.instance(n).classIndex()) {
                        error = 1 - m_OutputLayer.get(maxOutputIndex).getOutput();
                    } else {
                        error = 0 - m_OutputLayer.get(maxOutputIndex).getOutput();
                    }
                }
            }
             
            mse += Math.pow(error, 2);
            System.out.println("error: " + error);
        }
        mse *= 0.5;

        return mse;
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

    public Instances getInstances() {
        return m_Instances;
    }

    public void setInstances(Instances m_Instances) {
        this.m_Instances = m_Instances;
    }

    public int getMaxIteration() {
        return m_MaxIteration;
    }

    public void setMaxIteration(int m_MaxIteration) {
        this.m_MaxIteration = m_MaxIteration;
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

    public double getThreshold() {
        return m_Threshold;
    }

    public void setThreshold(double m_Threshold) {
        this.m_Threshold = m_Threshold;
    }

}
