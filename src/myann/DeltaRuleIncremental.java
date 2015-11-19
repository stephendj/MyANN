package myann;

import java.util.List;
import weka.core.Instances;


public class DeltaRuleIncremental extends SingleLayerPerceptron {
    private static final double THRESHOLD = 0.01;
    
    public DeltaRuleIncremental(int m_MaxIteration, double m_LearningRate, double m_Momentum, int numInput) {
        super(m_MaxIteration, new Neuron("no", numInput), m_LearningRate, m_Momentum);
    }
    
    public DeltaRuleIncremental(int m_MaxIteration, double m_LearningRate, 
            double m_Momentum, double biasWeight, List<Double> weights) {
        super(m_MaxIteration, new Neuron("no", biasWeight, weights), m_LearningRate, m_Momentum);
    }
    
    private void learning(Instances instances) {
        // 1 Epoch
        for (int i = 0; i < instances.numInstances(); ++i) {
            List<Double> deltaWeights = super.calculateDeltaWeight(i);
            double deltaBiasWeight = super.calculateDeltaBiasWeight(i);
            super.updateWeights(deltaWeights, deltaBiasWeight);
        }
    }
    
    private void print1Epoch(int epoch) {
        System.out.println("==================================");
        System.out.println("Model Pembelajaran Epoch " + epoch + "\n");
        super.printModel();
        System.out.println("==================================\n");
    }


    @Override
    public void buildClassifier(Instances instances) throws Exception {
        super.setInstances(instances);
        double mse = 1.0;
        int iteration = 1;
        if (super.getMaxIteration() == 0) {
            while (Double.compare(mse, THRESHOLD) > 0 && Double.compare(mse, Double.POSITIVE_INFINITY) < 0 ) {
                learning(instances);
                print1Epoch(iteration);
                mse = super.calculateMSE();
                System.out.println("MSE: "+mse);
                ++iteration;
            }
        } else { // stop if convergen before reaching max iteration
            while (iteration <= super.getMaxIteration() && Double.compare(mse, THRESHOLD) > 0 
                    && Double.compare(mse, Double.POSITIVE_INFINITY) < 0 ) {
                learning(instances);
                print1Epoch(iteration);
                mse = super.calculateMSE();
                System.out.println("MSE: "+mse);
                ++iteration;
            }
        }
        
    }
}
