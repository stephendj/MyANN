package myann;

import java.util.ArrayList;
import java.util.List;
import weka.core.Instance;
import weka.core.Instances;

public class PerceptronTrainingRule extends SingleLayerPerceptron {

    private static final double THRESHOLD = 0.01;

    /**
     *
     * @param m_MaxIteration maximum epoch iteration, 0 for not using maximum iteration
     * @param m_Neuron neuron
     * @param m_LearningRate learning rate [0..1]
     * @param m_Momentum momentum
     */
    public PerceptronTrainingRule(int m_MaxIteration, List<Neuron> m_Neuron, double m_LearningRate, double m_Momentum) {
        super(m_MaxIteration, m_Neuron, m_LearningRate, m_Momentum);
    }
    
//    public PerceptronTrainingRule(int m_MaxIteration, double m_LearningRate, 
//            double m_Momentum, double biasWeight, List<Double> weights, String activationFunction) {
//        super(m_MaxIteration, new Neuron(activationFunction, biasWeight, weights), m_LearningRate, m_Momentum);
//    }

    private void learning(Instances instances) {
        // 1 Epoch
        for (int i = 0; i < instances.numInstances(); ++i) {
            List<List<Double>> deltaWeights = super.calculateDeltaWeight(i);
            List<Double> deltaBiasWeight = super.calculateDeltaBiasWeight(i);
            super.updateWeights(deltaWeights, deltaBiasWeight);
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
        if (super.getMaxIteration() == 0) {
            while (Double.compare(error, THRESHOLD) > 0) {
                learning(instances);
                print1Epoch(iteration);
                error = super.calculateMSE();
                System.out.println("mse : " + error);
                ++iteration;
            }
        } else {
            while (iteration <= super.getMaxIteration() && Double.compare(error, THRESHOLD) > 0) {
                learning(instances);
                print1Epoch(iteration);
                error = super.calculateMSE();
                System.out.println("mse : " + error);
                ++iteration;
            }
        }
    }

}
