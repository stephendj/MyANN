package myann;

import myann.nominalconverter.NominalConverter;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class MyANN {

    private static final String DATASET1 = "data/weather.nominal.arff";
    
    public static void main(String[] args) {
        Instances instances = Helper.loadDataFromFile(DATASET1);
        Classifier classifierPTR = Helper.buildClassifier(instances, "dri");
//        System.out.println(NominalConverter.nominalToBinary(instances, false));
    }

}
