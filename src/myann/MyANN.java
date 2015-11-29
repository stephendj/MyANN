package myann;

import myann.nominalconverter.NominalConverter;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class MyANN {

    private static final String DATASET = "data/iris.arff";
    private static final String DATASET_UNLABELED = "data/iris.unlabeled.arff";
    
    public static void main(String[] args) throws Exception {
        Instances instances = Helper.loadDataFromFile(DATASET);
        Instances unlabeledInstances = Helper.loadDataFromFile(DATASET_UNLABELED);
        System.out.println(NominalConverter.nominalToNumeric(instances));
        System.out.println(NominalConverter.nominalToNumeric(unlabeledInstances));
        unlabeledInstances = NominalConverter.nominalToNumeric(unlabeledInstances);
        Classifier classifier = Helper.buildClassifier(instances, "mlp");
        for (int i=0; i<unlabeledInstances.numInstances(); ++i) {
            System.out.println(classifier.classifyInstance(unlabeledInstances.instance(i)));
        }
    }

}
