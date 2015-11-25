package myann.nominalconverter;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

public class NominalConverter {

    /**
     *
     * @param instances instances to be converted
     * @return new instances whose attributes have been converted to numeric
     */
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
    
    /**
     *
     * @param instances instances to be converted
     * @param compact true if not all attributes will be converted, otherwise false
     * @return new instances whose attributes have been converted
     */
    public static Instances nominalToBinary(Instances instances, boolean compact) {
        NominalToBinary nominalToBinaryFilter = new NominalToBinary();
        try {
            nominalToBinaryFilter.setInputFormat(instances);
            String[] options;
            if (compact) {
                options= new String[]{"A"};
            } else {
                options= new String[]{"-A"};
            }
            nominalToBinaryFilter.setOptions(options);
            return Filter.useFilter(instances, nominalToBinaryFilter);
        } catch(Exception e) {
            e.printStackTrace();
        }
        return instances; //Unable to convert
    }
}
