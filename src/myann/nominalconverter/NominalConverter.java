package myann.nominalconverter;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class NominalConverter {
    public static Instances nominalToNumeric(Instances instances) {
        FastVector newAttributes = new FastVector(instances.numAttributes());
        for(int i = 0; i < instances.numAttributes(); ++i) {
            if(!instances.attribute(i).equals(instances.classAttribute())) {
                newAttributes.addElement(new Attribute(instances.attribute(i).name()));
            } else {
                newAttributes.addElement(instances.classAttribute());
            }
        }
        
        Instances newInstances = new Instances(instances.relationName(), newAttributes, instances.numInstances());
        newInstances.setClassIndex(instances.classIndex());
        
        for(int i = 0; i < instances.numInstances(); ++i) {
            Instance newInstance = new Instance(newInstances.numAttributes());
            for(int j = 0; j < newInstances.numAttributes(); ++j) {
                newInstance.setValue(newInstances.attribute(j), instances.instance(i).value(j));
            }
            newInstances.add(newInstance);
        }
        
        return newInstances;
    }
}
