package myann;

import java.util.ArrayList;
import java.util.List;
import weka.core.Instances;

public class PerceptronTrainingRule extends SingleLayerPerceptron {

    private static final double INITIAL_DELTA_WEIGHT = 0.0;
    private static final double THRESHOLD = 0.01;

    private List<Double> deltaBiasWeightPrev; //update pter iterate
    private List<List<Double>> deltaWeightPrev; //update per iterate

    /**
     *
     * @param m_MaxIteration maximum epoch iteration, 0 for not using maximum
     * iteration
     * @param m_Neuron neuron
     * @param m_LearningRate learning rate [0..1]
     * @param m_Momentum momentum [0..1]
     */
    public PerceptronTrainingRule(int m_MaxIteration, List<Neuron> m_Neuron, double m_LearningRate, double m_Momentum) {
        super(m_MaxIteration, m_Neuron, m_LearningRate, m_Momentum);
    }

//    public PerceptronTrainingRule(int m_MaxIteration, double m_LearningRate, 
//            double m_Momentum, double biasWeight, List<Double> weights, String activationFunction) {
//        super(m_MaxIteration, new Neuron(activationFunction, biasWeight, weights), m_LearningRate, m_Momentum);
//    }
    
    private void deltaWeightInitiation() {
        deltaBiasWeightPrev = new ArrayList<>();
        for (int i = 0; i < getNeuron().size(); ++i) {
            deltaBiasWeightPrev.add(INITIAL_DELTA_WEIGHT);
        }
        deltaWeightPrev = new ArrayList<>();
        for (int i = 0; i < getNeuron().size(); ++i) {
            List<Double> deltaWeightPrevNeuron = new ArrayList<>();
            for (int j = 0; j < getNeuron().get(i).getWeight().size(); ++j) {
                deltaWeightPrevNeuron.add(INITIAL_DELTA_WEIGHT);
            }
            deltaWeightPrev.add(deltaWeightPrevNeuron);
        }
    }

    private void learning(Instances instances) {
        //learning 1 epoch
        for (int i = 0; i < instances.numInstances(); ++i) {
            List<List<Double>> deltaWeights = super.calculateDeltaWeight(i, deltaWeightPrev);
            deltaWeightPrev = deltaWeights;
            List<Double> deltaBiasWeight = super.calculateDeltaBiasWeight(i, deltaBiasWeightPrev);
            deltaBiasWeightPrev = deltaBiasWeight;
            super.updateWeights(deltaWeights, deltaBiasWeight);
            
//            for (int j = 0; j < deltaWeights.size(); j++) {
//                System.out.println("neuron : "+ j);
//                System.out.println("delta bias weight: "+ deltaBiasWeight.get(j));
//                for (int k = 0; k < deltaWeights.get(j).size(); k++) {
//                    System.out.println("delta w"+k+ ": "+deltaWeights.get(j).get(k));
//                }
//                
//            }
        }
    }

    private void print1Epoch(int epoch) {
        System.out.println("==================================");
        System.out.println("Model Pembelajaran Epoch " + epoch + "\n");
        super.printModel();
        System.out.println("==================================\n");
    }

    /**
     *
     * @param instances data train
     * @throws Exception if classifier failed to build
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        super.setInstances(instances);
        double error = 1.0;
        int iteration = 1;

        deltaWeightInitiation();
        if (super.getMaxIteration() == 0) {
            while (Double.compare(error, THRESHOLD) > 0) {
                learning(instances);
                print1Epoch(iteration);
                error = super.calculateMSE();
                System.out.println("mse: " + error);
                ++iteration;
            }
        } else {
            while (iteration <= super.getMaxIteration() && Double.compare(error, THRESHOLD) > 0) {
                learning(instances);
                print1Epoch(iteration);
                error = super.calculateMSE();
                System.out.println("mse: " + error);
                ++iteration;
            }
        }
    }

}
