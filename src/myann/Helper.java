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

    private static String method;
    private static String isNormalize;

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
            e.printStackTrace();
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

    private static double inputWeightBias(boolean isRandomize) {
        if (isRandomize) {
            Random random = new Random();
            return random.nextDouble();
        } else {
            Scanner in = new Scanner(System.in);
            System.out.print("Masukkan nilai weight bias : ");
            return in.nextDouble();
        }
    }

    private static List<Double> inputWeights(int numAttributes, int neuronIndex, boolean isRandomize) {
        List<Double> weights = new ArrayList<>();
        if (isRandomize) {
            Random random = new Random();
            for (int j = 0; j < numAttributes - 1; ++j) {
                weights.add(random.nextDouble());
            }
        } else {
            Scanner in = new Scanner(System.in);

            for (int j = 0; j < numAttributes - 1; ++j) {
                if (neuronIndex == -1) {
                    System.out.print("Masukkan w" + j + " : ");
                } else {
                    System.out.print("Masukkan w" + j + neuronIndex + " : ");
                }
                double inputWeight = in.nextDouble();
                weights.add(inputWeight);
            }
        }
        return weights;
    }

    private static Neuron inputNeuron(String activationFunction, double biasWeight, List<Double> weights) {
        switch (activationFunction) {
            case "sign":
                return new Neuron(ActivationFunction.SIGN, biasWeight, weights);
            case "step":
                return new Neuron(ActivationFunction.STEP, biasWeight, weights);
            case "sigmoid":
                return new Neuron(ActivationFunction.SIGMOID, biasWeight, weights);
            default:
                return new Neuron(ActivationFunction.NONE, biasWeight, weights);
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
            Random random = new Random();

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
            method = in.nextLine();
            switch (method) {
                case "numeric":
                    data = NominalConverter.nominalToNumeric(data);
                    break;
                case "binary_full":
                    data = NominalConverter.nominalToBinary(data, false);
                    break;
                case "binary_partial":
                    data = NominalConverter.nominalToBinary(data, true);
                    break;
                default:
                    break;
            }

            System.out.print("Apakah data ingin dinormalisasi (y/n) ? ");
            isNormalize = in.nextLine();
            switch (isNormalize) {
                case "y":
                    Normalize normalize = new Normalize();
                    normalize.setInputFormat(data);
                    data = Filter.useFilter(data, normalize);
                    break;
                default:
                    break;
            }
            
            System.out.println(data);

            System.out.print("Apakah weight ingin diinisialisasi (y/n) ? ");
            boolean isRandomize = in.nextLine().equals("y") ? false : true;

//            int maxIteration = 10000;
//            double learningRate = 0.1;
//            double momentum = 0.1;
//            double threshold = 0.01;
            switch (type.toLowerCase()) {
                case "ptr":
                    List<Neuron> neurons = new ArrayList<>();

                    if (data.classAttribute().numValues() > 2) { // Multi class
                        for (int i = 0; i < data.classAttribute().numValues(); ++i) {
                            String activationFunction = inputActivationFunction();
                            double biasWeight = inputWeightBias(isRandomize);
                            List<Double> weights = inputWeights(data.numAttributes(), i, isRandomize);
                            neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                        }
                    } else { // Binary Class
                        String activationFunction = inputActivationFunction();
                        double biasWeight = inputWeightBias(isRandomize);
                        List<Double> weights = inputWeights(data.numAttributes(), -1, isRandomize);
                        neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                    }

                    PerceptronTrainingRule PTR = new PerceptronTrainingRule(maxIteration,
                        neurons, learningRate, momentum, threshold);
                    System.out.println(data);
                    PTR.buildClassifier(data);
                    
                    Helper.tenFoldCrossValidation(data, PTR);

                    return PTR;

                case "drb":
                    neurons = new ArrayList<>();

                    if (data.classAttribute().numValues() > 2) { // Multi class
                        for (int i = 0; i < data.classAttribute().numValues(); ++i) {
                            String activationFunction = "none";
                            double biasWeight = inputWeightBias(isRandomize);
                            List<Double> weights = inputWeights(data.numAttributes(), i, isRandomize);
                            neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                        }
                    } else { // Binary Class
                        String activationFunction = "none";
                        double biasWeight = inputWeightBias(isRandomize);
                        List<Double> weights = inputWeights(data.numAttributes(), -1, isRandomize);
                        neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                    }

                    DeltaRuleBatch DRB = new DeltaRuleBatch(maxIteration,
                        neurons, learningRate, momentum, threshold);
                    DRB.buildClassifier(data);
                    
                    Helper.tenFoldCrossValidation(data, DRB);

                    return DRB;

                case "dri":
                    neurons = new ArrayList<>();

                    if (data.classAttribute().numValues() > 2) { // Multi class
                        for (int i = 0; i < data.classAttribute().numValues(); ++i) {
                            String activationFunction = "none";
                            double biasWeight = inputWeightBias(isRandomize);
                            List<Double> weights = inputWeights(data.numAttributes(), i, isRandomize);
                            neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                        }
                    } else { // Binary Class
                        String activationFunction = "none";
                        double biasWeight = inputWeightBias(isRandomize);
                        List<Double> weights = inputWeights(data.numAttributes(), -1, isRandomize);
                        neurons.add(inputNeuron(activationFunction, biasWeight, weights));
                    }

                    DeltaRuleIncremental DRI = new DeltaRuleIncremental(maxIteration,
                        neurons, learningRate, momentum, threshold);
                    DRI.buildClassifier(data);

                    Helper.tenFoldCrossValidation(data, DRI);
                    return DRI;

                case "mlp":
                    List<List<Neuron>> hiddenLayers = new ArrayList<>();
                    List<Neuron> outputLayer = new ArrayList<>();

                    // read hidden layers weights from stdin
                    System.out.print("Masukkan jumlah layer hidden layer : ");
                    int jumlahLayer = in.nextInt();

                    for (int i = 0; i < jumlahLayer; ++i) {
                        System.out.print("Masukkan jumlah neuron hidden layer " + i + " : ");
                        int jumlahSetiapLayer = in.nextInt();

                        List<Neuron> hiddenLayer = new ArrayList<>();
                        for (int j = 0; j < jumlahSetiapLayer; ++j) {
                            double biasWeight = inputWeightBias(isRandomize);
                            List<Double> inputWeights = inputWeights(data.numAttributes(), jumlahSetiapLayer > 1 ? j : -1, isRandomize);
                            hiddenLayer.add(inputNeuron(ActivationFunction.SIGMOID, biasWeight, inputWeights));
                        }
                        hiddenLayers.add(hiddenLayer);
                    }

                    // read output layer weights from stdin
                    System.out.println("Jumlah neuron output layer : " + ((data.classAttribute().numValues() > 2) ? data.classAttribute().numValues() : 1));
                    if (data.classAttribute().numValues() > 2) { // multiclass
                        for (int i = 0; i < data.classAttribute().numValues(); ++i) {
                            double biasWeight = inputWeightBias(isRandomize);
                            List<Double> outputWeights = inputWeights(hiddenLayers.get(hiddenLayers.size() - 1).size() + 1, i, isRandomize);
                            outputLayer.add(inputNeuron(ActivationFunction.SIGMOID, biasWeight, outputWeights));
                        }
                    } else { // binary class
                        double biasWeight = inputWeightBias(isRandomize);
                        List<Double> outputWeights = inputWeights(hiddenLayers.get(hiddenLayers.size() - 1).size() + 1, -1, isRandomize);
                        outputLayer.add(inputNeuron(ActivationFunction.SIGMOID, biasWeight, outputWeights));
                    }

                    MultiLayerPerceptron MLP = new MultiLayerPerceptron(hiddenLayers, outputLayer, data, maxIteration,
                        learningRate, momentum, threshold);
                    MLP.buildClassifier(data);
                    
                    Helper.tenFoldCrossValidation(data, MLP);
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
            e.printStackTrace();
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
            e.printStackTrace();
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
            e.printStackTrace();
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
            e.printStackTrace();
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
            e.printStackTrace();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
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

            switch (method) {
                case "numeric":
                    unlabeled = NominalConverter.nominalToNumeric(unlabeled);
                    break;
                case "binary_full":
                    unlabeled = NominalConverter.nominalToBinary(unlabeled, false);
                    break;
                case "binary_partial":
                    unlabeled = NominalConverter.nominalToBinary(unlabeled, true);
                    break;
                default:
                    break;
            }
            
            switch (isNormalize) {
                case "y":
                    Normalize normalize = new Normalize();
                    normalize.setInputFormat(unlabeled);
                    unlabeled = Filter.useFilter(unlabeled, normalize);
                    break;
                default:
                    break;
            }
            
            Instances labeled = new Instances(unlabeled);

            // label instances
            for (int i = 0; i < unlabeled.numInstances(); i++) {
                double clsLabel = classifier.classifyInstance(unlabeled.instance(i));
                labeled.instance(i).setClassValue(clsLabel);
                System.out.println(labeled.instance(i));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
