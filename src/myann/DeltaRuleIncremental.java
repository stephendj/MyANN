package myann;

import java.util.ArrayList;
import java.util.List;
import weka.core.Instances;

public class DeltaRuleIncremental extends SingleLayerPerceptron {

    private static final double INITIAL_DELTA_WEIGHT = 0.0;

    private List<Double> deltaBiasWeightPrev; //update per iterate
    private List<List<Double>> deltaWeightPrev; // update per iterate
    private double m_Threshold;

    /**
     *
     * @param m_MaxIteration maximum epoch iteration, 0 for not using maximum iteration
     * @param m_Neuron neuron
     * @param m_LearningRate learning rate [0..1]
     * @param m_Momentum momentum [0..1]
     */
    public DeltaRuleIncremental(int m_MaxIteration, List<Neuron> m_Neuron, double m_LearningRate, double m_Momentum, double m_Threshold) {
        super(m_MaxIteration, m_Neuron, m_LearningRate, m_Momentum);
        this.m_Threshold = m_Threshold;
    }

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
        // 1 Epoch
        for (int i = 0; i < instances.numInstances(); ++i) {
            List<List<Double>> deltaWeights = super.calculateDeltaWeight(i, deltaWeightPrev);
            deltaWeightPrev = deltaWeights;
            List<Double> deltaBiasWeight = super.calculateDeltaBiasWeight(i, deltaBiasWeightPrev);
            deltaBiasWeightPrev = deltaBiasWeight;
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
        double mse = 1.0;
        int iteration = 1;

        deltaWeightInitiation();
        if (super.getMaxIteration() == 0) {
            while (Double.compare(mse, m_Threshold) > 0 && Double.compare(mse, Double.POSITIVE_INFINITY) < 0) {
                learning(instances);
                print1Epoch(iteration);
                mse = super.calculateMSE();
                System.out.println("MSE: " + mse);
                ++iteration;
            }
        } else { // stop if convergen before reaching max iteration
            while (iteration <= super.getMaxIteration() && Double.compare(mse, m_Threshold) > 0
                    && Double.compare(mse, Double.POSITIVE_INFINITY) < 0) {
                learning(instances);
                print1Epoch(iteration);
                mse = super.calculateMSE();
                System.out.println("MSE: " + mse);
                ++iteration;
            }
        }

    }
}
