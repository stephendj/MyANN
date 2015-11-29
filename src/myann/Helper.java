package myann;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import myann.activationfunction.ActivationFunction;
import myann.nominalconverter.NominalConverter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

public class Helper {

    /**
     * Constructor
     */
    public Helper() {
    }

    /**
     * Load the dataset from specified file into weka Instances
     *
     * @param file the file path to the dataset
     * @return
     */
    public static Instances loadDataFromFile(String file) {
        Instances data = null;

        try {
            data = DataSource.read(file);

            // setting class attribute if the data format does not provide this information
            // For example, the ARFF format saves the class attribute information as well
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
        } catch (Exception e) {
        }

        return data;
    }

    /**
     * Remove the attributes from the dataset
     *
     * @param data
     * @param attribute a string representing the list of attributes. Since the
     * string will typically come from a user, attributes are indexed from 1.
     * eg: first-3,5,6-last
     * @return
     */
    public static Instances removeAttribute(Instances data, String attribute) {
        Instances newData = null;

        try {
            Remove remove = new Remove();
            remove.setAttributeIndices(attribute);
            remove.setInputFormat(data);
            newData = Filter.useFilter(data, remove);
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }

        return newData;
    }

    /**
     * Randomize the instances
     *
     * @param data the dataset that will be randomized
     * @return
     */
    public static Instances resample(Instances data) {
        Instances newData = null;

        try {
            Resample resample = new Resample();
            resample.setInputFormat(data);
            newData = Filter.useFilter(data, resample);
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }

        return newData;
    }

    private static String inputActivationFunction() {
        Scanner in = new Scanner(System.in);
        System.out.print("Masukkan fungsi aktivasi (sign, step, sigmoid, none) : ");
        return in.nextLine();
    }
    
    private static double inputWeightBias() {
        Scanner in = new Scanner(System.in);
        System.out.print("Masukkan nilai weight bias : ");
        return in.nextDouble();
    }
    
    private static List<Double> inputWeights(int numAttributes, int neuronIndex) {
        Scanner in = new Scanner(System.in);
        List<Double> weights = new ArrayList<>();
        for(int j = 0; j < numAttributes - 1; ++j) {
            if(neuronIndex == -1) {
                System.out.print("Masukkan w" + j + " : ");
            } else {
                System.out.print("Masukkan w" + j + neuronIndex + " : ");
            }
            double inputWeight = in.nextDouble();
            weights.add(inputWeight);
        }
        return weights;
    }
    
    private static Neuron inputNeuron(String activationFunction, double biasWeight, List<Double> weights) {
        switch(activationFunction) {
            case "sign"    : return new Neuron(ActivationFunction.SIGN, biasWeight, weights);
            case "step"    : return new Neuron(ActivationFunction.STEP, biasWeight, weights);
            case "sigmoid" : return new Neuron(ActivationFunction.SIGMOID, biasWeight, weights);
            default        : return new Neuron(ActivationFunction.NONE, biasWeight, weights);
        }
    }
    
    /**
     * Build the classifier from dataset, allowed algorithms are perceptron
     * training rule, delta rule batch, delta rule incremental, multi layer
     * perceptron
     *
     * @param data the dataset that will be trained
     * @param type choice of algorithm, can be perceptron training rule, delta
     * rule batch, delta rule incremental, or multi layer perceptron
     * @return
     */
    public static Classifier buildClassifier(Instances data, String type) {
        try {
            // Input user
            Scanner in = new Scanner(System.in);
            
            System.out.print("Masukkan jumlah iterasi (0 jika tidak ada maks) : "); 
            int maxIteration = in.nextInt();
            
            System.out.print("Masukkan learning rate : ");
            double learningRate = in.nextDouble();
            
            System.out.print("Masukkan momentum : ");
            double momentum = in.nextDouble();
            
            System.out.print("Masukkan threshold : ");
            double threshold = in.nextDouble();
            
            System.out.print("Pilih metode convert data (numeric, binary_full, binary_partial, none) : ");
            in.nextLine();
            String method = in.nextLine();
            switch(method) {
                case "numeric"        : data = NominalConverter.nominalToNumeric(data); break;
                case "binary_full"    : data = NominalConverter.nominalToBinary(data, true); break;
                case "binary_partial" : data = NominalConverter.nominalToBinary(data, false); break;
                default               : break;
            }
            
            System.out.print("Apakah data ingin dinormalisasi (y/n) ? ");
            String isNormalize = in.nextLine();
            switch(isNormalize) {
                case "y" : Normalize normalize = new Normalize();
                           normalize.setInputFormat(data);
                           data = Filter.useFilter(data, normalize);
                           break;
                default  : break;
            }
            
            switch (type.toLowerCase()) {
                case "ptr":
                    List<Neuron> neurons = new ArrayList<>();
                    
                    if(data.classAttribute().numValues() > 2) { // Multi class
                        for(int i = 0; i < data.classAttribute().numValues(); ++i) {
                            String activationFunction = inputActivationFunction();
                            double biasWeight = inputWeightBias();
                            List<Double> weights = inputWeights(data.numAttributes(), i);
                            neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                        }
                    } else { // Binary Class
                        String activationFunction = inputActivationFunction();
                        double biasWeight = inputWeightBias();
                        List<Double> weights = inputWeights(data.numAttributes(), -1);
                        neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                    }
                    
                    PerceptronTrainingRule PTR = new PerceptronTrainingRule(maxIteration,
                        neurons, learningRate, momentum, threshold);
                    PTR.buildClassifier(data);

                    return PTR;

                case "drb":
                    neurons = new ArrayList<>();
                    
                    if(data.classAttribute().numValues() > 2) { // Multi class
                        for(int i = 0; i < data.classAttribute().numValues(); ++i) {
                            String activationFunction = "none";
                            double biasWeight = inputWeightBias();
                            List<Double> weights = inputWeights(data.numAttributes(), i);
                            neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                        }
                    } else { // Binary Class
                        String activationFunction = "none";
                        double biasWeight = inputWeightBias();
                        List<Double> weights = inputWeights(data.numAttributes(), -1);
                        neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                    }
                    
                    DeltaRuleBatch DRB = new DeltaRuleBatch(maxIteration,
                        neurons, learningRate, momentum, threshold);
                    DRB.buildClassifier(data);

                    return DRB;

                case "dri":
                    neurons = new ArrayList<>();

                    if(data.classAttribute().numValues() > 2) { // Multi class
                        for(int i = 0; i < data.classAttribute().numValues(); ++i) {
                            String activationFunction = "none";
                            double biasWeight = inputWeightBias();
                            List<Double> weights = inputWeights(data.numAttributes(), i);
                            neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                        }
                    } else { // Binary Class
                        String activationFunction = "none";
                        double biasWeight = inputWeightBias();
                        List<Double> weights = inputWeights(data.numAttributes(), -1);
                        neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                    }
                    
                    DeltaRuleIncremental DRI = new DeltaRuleIncremental(maxIteration,
                        neurons, learningRate, momentum, threshold);
                    DRI.buildClassifier(data);

                    return DRI;

                case "mlp":
                    List<Double> inputWeights1 = new ArrayList<>();
                    inputWeights1.add(0.0);
                    inputWeights1.add(0.0);
                    inputWeights1.add(0.0);
                    inputWeights1.add(0.0);
                    List<Double> inputWeights2 = new ArrayList<>();
                    inputWeights2.add(0.0);
                    inputWeights2.add(0.0);
                    inputWeights2.add(0.0);
                    inputWeights2.add(0.0);

                    Neuron hiddenNeuron1 = new Neuron(ActivationFunction.SIGMOID, 0, inputWeights1);
                    Neuron hiddenNeuron2 = new Neuron(ActivationFunction.SIGMOID, 0, inputWeights2);
                    List<Neuron> hiddenLayer1 = new ArrayList<>();
                    hiddenLayer1.add(hiddenNeuron1);
                    hiddenLayer1.add(hiddenNeuron2);

                    List<List<Neuron>> hiddenLayers = new ArrayList<>();
                    hiddenLayers.add(hiddenLayer1);

                    List<Double> outputWeights = new ArrayList<>();
                    outputWeights.add(0.0);
                    outputWeights.add(0.0);

                    Neuron outputNeuron = new Neuron(ActivationFunction.SIGMOID, 0, outputWeights);
                    List<Neuron> outputLayer = new ArrayList<>();
                    outputLayer.add(outputNeuron);

                    MultiLayerPerceptron MLP = new MultiLayerPerceptron(hiddenLayers, outputLayer, data, maxIteration,
                        learningRate, momentum, threshold);
                    MLP.buildClassifier(data);

                    return MLP;
            }
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }

        return null;
    }

    /**
     * Do a ten fold cross validation using he model and instances
     *
     * @param data the dataset that will be used
     * @param classifier the classifier that will be used
     */
    public static void tenFoldCrossValidation(Instances data,
        Classifier classifier) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data,
                10, new Random(1));
            System.out
                .println(eval.toSummaryString("=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
        }
    }

    /**
     * Function to test the classifier that has been built
     *
     * @param data the training set that will be used
     * @param classifier the classifier that will be used
     * @param datatest the test set that will be used
     */
    public static void testSetEvaluation(Instances data, Classifier classifier, Instances datatest) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(classifier, datatest);

            System.out
                .println(eval.toSummaryString("=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            // TODO Auto-generated catch block
        }
    }

    /**
     * Split instances into training data and test data to test the classifier
     *
     * @param data the dataset that will be split
     * @param classifier the classifier that will be used
     * @param percentage the percentage of the split, usually 80 which means 80%
     */
    public static void percentageSplit(Instances data, Classifier classifier, int percentage) {
        Instances dataSet = new Instances(data);
        dataSet.randomize(new Random(1));

        int trainSize = Math.round(dataSet.numInstances() * percentage / 100);
        int testSize = dataSet.numInstances() - trainSize;
        Instances trainSet = new Instances(dataSet, 0, trainSize);
        Instances testSet = new Instances(dataSet, trainSize, testSize);

        try {
            classifier.buildClassifier(trainSet);
            testSetEvaluation(trainSet, classifier, testSet);
        } catch (Exception e) {
        }
    }

    /**
     * Save a model that has been built to a file
     *
     * @param classifier the classifier that will be saved
     * @param file the filename for the file
     */
    public static void saveModelToFile(Classifier classifier, String file) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(file));
            oos.writeObject(classifier);

            oos.flush();
            oos.close();
        } catch (Exception e) {
        }
    }

    /**
     * Load a classifier model from file
     *
     * @param file the file path to the model file
     * @return
     */
    public static Classifier loadModelFromFile(String file) {
        Classifier cls = null;

        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
                file));
            cls = (Classifier) ois.readObject();

            ois.close();
        } catch (FileNotFoundException e) {
        } catch (IOException | ClassNotFoundException e) {
        }

        return cls;
    }

    /**
     * Classify unlabeled instances in a file using a classifier
     *
     * @param classifier the classifier chosen to classify the instances
     * @param file the file path to the unlabeled instances
     */
    public static void classifyUsingModel(Classifier classifier, String file) {
        try {
            Instances unlabeled = DataSource.read(file);
            unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

            Instances labeled = new Instances(unlabeled);

            // label instances
            for (int i = 0; i < unlabeled.numInstances(); i++) {
                double clsLabel = classifier.classifyInstance(unlabeled.instance(i));
                labeled.instance(i).setClassValue(clsLabel);
                System.out.println(labeled.instance(i));
            }
        } catch (Exception e) {
        }
    }
}
