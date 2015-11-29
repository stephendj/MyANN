package myann;

import myann.nominalconverter.NominalConverter;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class MyANN {

    private static final String DATASET = "data/weather.numeric.arff";
    private static final String DATASET_UNLABELED = "data/weather.numeric.unlabeled.arff";
    
    public static void main(String[] args) throws Exception {
        Instances instances = Helper.loadDataFromFile(DATASET);
        
        Classifier classifier = Helper.buildClassifier(instances, "dri");
        Helper.classifyUsingModel(classifier, DATASET_UNLABELED);
    }

}
