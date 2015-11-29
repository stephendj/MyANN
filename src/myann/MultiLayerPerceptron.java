package myann;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;

public class MultiLayerPerceptron extends Classifier {

    private List<List<Neuron>> m_HiddenLayer = new ArrayList<>();
    private List<Neuron> m_OutputLayer = new ArrayList<>();
    private Instances m_Instances;
    private int m_MaxIteration;
    private double m_LearningRate;
    private double m_Momentum;
    private double m_Threshold;

    private List<List<List<Double>>> prevHiddenDeltaWeights = new ArrayList<>();
    private List<List<Double>> prevHiddenDeltaBiasWeights = new ArrayList<>();
    private List<List<Double>> prevOutputDeltaWeights = new ArrayList<>();
    private List<Double> prevOutputDeltaBiasWeights = new ArrayList<>();

    public MultiLayerPerceptron(List<List<Neuron>> hiddenLayer,
        List<Neuron> outputLayer, Instances m_Instances, int m_MaxIteration, double m_LearningRate, double m_Momentum, double m_Threshold) {
        this.m_HiddenLayer = hiddenLayer;
        this.m_OutputLayer = outputLayer;
        this.m_Instances = m_Instances;
        this.m_MaxIteration = m_MaxIteration;
        this.m_LearningRate = m_LearningRate;
        this.m_Momentum = m_Momentum;
        this.m_Threshold = m_Threshold;

        // initialize previous delta weights and previous delta bias weights
        for (int i = 0; i < m_HiddenLayer.size(); ++i) {
            List<List<Double>> tHiddenDeltaWeights = new ArrayList<>();
            for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                List<Double> ttHiddenDeltaWeights = new ArrayList<>();
                for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); ++k) {
                    ttHiddenDeltaWeights.add(0.0);
                }
                tHiddenDeltaWeights.add(ttHiddenDeltaWeights);
            }
            prevHiddenDeltaWeights.add(tHiddenDeltaWeights);
        }

        for (int i = 0; i < m_HiddenLayer.size(); ++i) {
            List<Double> tHiddenDeltaBiasWeights = new ArrayList<>();
            for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                tHiddenDeltaBiasWeights.add(0.0);
            }
            prevHiddenDeltaBiasWeights.add(tHiddenDeltaBiasWeights);
        }

        for (int i = 0; i < m_OutputLayer.size(); ++i) {
            List<Double> tOutputDeltaWeights = new ArrayList<>();
            for (int j = 0; j < m_OutputLayer.get(i).getWeight().size(); ++j) {
                tOutputDeltaWeights.add(0.0);
            }
            prevOutputDeltaWeights.add(tOutputDeltaWeights);
        }

        for (int i = 0; i < m_OutputLayer.size(); ++i) {
            prevOutputDeltaBiasWeights.add(0.0);
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        double mse = Double.MAX_VALUE;
        int epoch = 1;
        while (m_MaxIteration == 0 && mse > m_Threshold
            || mse > m_Threshold && epoch <= m_MaxIteration
            || Double.compare(mse, Double.POSITIVE_INFINITY) >= 0) {

            for (int i = 0; i < instances.numInstances(); ++i) {
                forwardChaining(instances.instance(i));
                backwardPropagation(instances.instance(i));
            }

            print1Epoch(epoch);
            mse = calculateMSE();
            System.out.println("MSE : " + mse);
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
                    List<Double> inputs = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i - 1).size(); ++k) { // previous hidden layer
                        inputs.add(m_HiddenLayer.get(i - 1).get(k).getOutput());
                    }
                    m_HiddenLayer.get(i).get(j).calculateOutput(inputs);
                }
            }
        }

        // output layer
        for (int j = 0; j < m_OutputLayer.size(); ++j) {
            List<Double> input = new ArrayList<>();
            for (int k = 0; k < m_HiddenLayer.get(m_HiddenLayer.size() - 1).size(); ++k) { // last hidden layer
                input.add(m_HiddenLayer.get(m_HiddenLayer.size() - 1).get(k).getOutput());
            }
            m_OutputLayer.get(j).calculateOutput(input);
        }
    }

    public void backwardPropagation(Instance instance) {
        List<Double> errorNext = new ArrayList<>();
        List<Double> errorNow = new ArrayList<>();

        // calculate error
        if (instance.classAttribute().isNumeric() || instance.classAttribute().numValues() == 2) {
            double error = m_OutputLayer.get(0).getOutput() * (1 - m_OutputLayer.get(0).getOutput()) * (instance.classValue() - m_OutputLayer.get(0).getOutput());
            errorNow.add(error);
        } else { // nominal multiclass
            double error;
            for (int i = 0; i < m_OutputLayer.size(); ++i) {
                if (Double.compare(i, instance.classValue()) == 0) {
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
                double deltaWeight = m_LearningRate * errorNow.get(j) * m_HiddenLayer.get(m_HiddenLayer.size() - 1).get(k).getOutput()
                    + m_Momentum * prevOutputDeltaWeights.get(j).get(k);
                double newWeight = m_OutputLayer.get(j).getWeight().get(k)
                    + deltaWeight;
                newWeights.add(newWeight);
                prevOutputDeltaWeights.get(j).set(k, deltaWeight);
            }
            m_OutputLayer.get(j).setWeight(newWeights);

            // update biasWeight outputLayer
            double deltaBiasWeight = m_LearningRate * errorNow.get(j) * m_OutputLayer.get(j).getBias()
                + m_Momentum * prevOutputDeltaBiasWeights.get(j);
            double newBiasWeight = m_OutputLayer.get(j).getBiasWeight()
                + deltaBiasWeight;
            m_OutputLayer.get(j).setBiasWeight(newBiasWeight);
            prevOutputDeltaBiasWeights.set(j, deltaBiasWeight);

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

                    // update weights
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                        double deltaWeight = m_LearningRate * error * instance.value(k)
                            + m_Momentum * prevHiddenDeltaWeights.get(i).get(j).get(k);
                        double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k)
                            + deltaWeight;
                        newWeights.add(newWeight);
                        prevHiddenDeltaWeights.get(i).get(j).set(k, deltaWeight);
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

                    // update weights
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                        double deltaWeight = m_LearningRate * error * m_HiddenLayer.get(i - 1).get(k).getOutput()
                            + m_Momentum * prevHiddenDeltaWeights.get(i).get(j).get(k);
                        double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k)
                            + deltaWeight;
                        newWeights.add(newWeight);
                        prevHiddenDeltaWeights.get(i).get(j).set(k, deltaWeight);
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

                    // update weights
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                        double deltaWeight = m_LearningRate * error * instance.value(k)
                            + m_Momentum * prevHiddenDeltaWeights.get(i).get(j).get(k);
                        double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k)
                            + deltaWeight;
                        newWeights.add(newWeight);
                        prevHiddenDeltaWeights.get(i).get(j).set(k, deltaWeight);
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

                    // update weights
                    List<Double> newWeights = new ArrayList<>();
                    for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); k++) {
                        double deltaWeight = m_LearningRate * error * m_HiddenLayer.get(i - 1).get(k).getOutput()
                            + m_Momentum * prevHiddenDeltaWeights.get(i).get(j).get(k);
                        double newWeight = m_HiddenLayer.get(i).get(j).getWeight().get(k)
                            + deltaWeight;
                        newWeights.add(newWeight);
                        prevHiddenDeltaWeights.get(i).get(j).set(k, deltaWeight);
                    }

                    // print newWeights
                    printNewWeights(newWeights);

                    m_HiddenLayer.get(i).get(j).setWeight(newWeights);
                }
            }

            // update biasWeight
            for (int j = 0; j < m_HiddenLayer.get(i).size(); j++) {
                double deltaBiasWeight = m_LearningRate * errorNow.get(j) * m_HiddenLayer.get(i).get(j).getBias()
                    + m_Momentum * prevHiddenDeltaBiasWeights.get(i).get(j);
                double newBiasWeight = m_HiddenLayer.get(i).get(j).getBiasWeight()
                    + deltaBiasWeight;
                m_HiddenLayer.get(i).get(j).setBiasWeight(newBiasWeight);
                prevHiddenDeltaBiasWeights.get(i).set(j, deltaBiasWeight);

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
            if (Double.compare(m_OutputLayer.get(i).getOutput(), m_OutputLayer.get(maxOutputIndex).getOutput()) > 0) {
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
                    if (Double.compare(i, m_Instances.instance(n).classValue()) == 0) {
                        error = 1 - m_OutputLayer.get(maxOutputIndex).getOutput();
                    } else {
                        error = 0 - m_OutputLayer.get(maxOutputIndex).getOutput();
                    }
                }
            }

            mse += Math.pow(error, 2);
        }
        mse *= 0.5;

        return mse;
    }

    /**
     *
     * @param instance the instance to be classified
     * @return the classification
     * @throws NoSupportForMissingValuesException
     */
    @Override
    public double classifyInstance(Instance instance)
        throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Multi Layer Perceptron: Cannot handle missing values");
        }

        forwardChaining(instance);

        // search for maximum output index
        int maxOutputIndex = 0;
        for (int i = 1; i < m_OutputLayer.size(); ++i) {
            if (Double.compare(m_OutputLayer.get(i).getOutput(), m_OutputLayer.get(maxOutputIndex).getOutput()) > 0) {
                maxOutputIndex = i;
            }
        }
        System.out.println("maxoutputindex " + maxOutputIndex);

        // classify instance
        if (instance.classAttribute().isNumeric()) {
            return m_OutputLayer.get(maxOutputIndex).getOutput();
        } else if (instance.classAttribute().numValues() == 2) { // nominal binary
            if (Double.compare(m_OutputLayer.get(maxOutputIndex).getOutput(), 0.5) >= 0) {
                return m_OutputLayer.get(maxOutputIndex).getOutput();
            } else {
                return m_OutputLayer.get(maxOutputIndex).getOutput();
            }
        } else { // nominal multiclass
            return maxOutputIndex;
        }
    }

    private void print1Epoch(int epoch) {
        System.out.println("==================================");
        System.out.println("Model Pembelajaran Epoch " + epoch + "\n");
        printModel();
        System.out.println("==================================\n");
    }

    /**
     * print the model of neuron
     */
    public void printModel() {
        for (int i = 0; i < m_HiddenLayer.size(); ++i) {
            for (int j = 0; j < m_HiddenLayer.get(i).size(); ++j) {
                System.out.println("Hidden Layer " + i + ", Neuron " + j);
                System.out.println("w_bias : " + m_HiddenLayer.get(i).get(j).getBiasWeight());
                for (int k = 0; k < m_HiddenLayer.get(i).get(j).getWeight().size(); ++k) {
                    System.out.println("w" + k + " : " + m_HiddenLayer.get(i).get(j).getWeight().get(k));
                }
            }
        }

        for (int i = 0; i < m_OutputLayer.size(); ++i) {
            System.out.println("Output Layer Neuron " + i);
            System.out.println("w_bias : " + m_OutputLayer.get(i).getBiasWeight());
            for (int j = 0; j < m_OutputLayer.get(i).getWeight().size(); ++j) {
                System.out.println("w" + j + " : " + m_OutputLayer.get(i).getWeight().get(j));
            }
        }
    }

    public void printNewBiasWeight(double newBiasWeight) {
//        System.out.println("newBiasWeight: " + newBiasWeight);
    }

    public void printNewWeights(List<Double> newWeights) {
//        for (int i = 0; i < newWeights.size(); ++i) {
//            System.out.println("newWeights " + i + ": " + newWeights.get(i));
//        }
    }

    public void printErrorNow(List<Double> errorNow) {
//        for (int i = 0; i < errorNow.size(); ++i) {
//            System.out.println("errorNow " + i + ": " + errorNow.get(i));
//        }
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
