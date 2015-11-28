package myann;

import java.util.ArrayList;
import java.util.List;
import weka.core.Instances;

public class DeltaRuleBatch extends SingleLayerPerceptron {

    private static final double INITIAL_DELTA_WEIGHT = 0.0;
    private static final double THRESHOLD = 0.01;

    private List<Double> deltaBiasWeightPrev;  //update per epoch
    private List<List<Double>> deltaWeightPrev; //update per epoch

    /**
     *
     * @param m_MaxIteration maximum epoch iteration, 0 for not using maximum iteration
     * @param m_Neuron neuron
     * @param m_LearningRate learning rate [0..1]
     * @param m_Momentum momentum [0..1]
     */
    public DeltaRuleBatch(int m_MaxIteration, List<Neuron> m_Neuron, double m_LearningRate, double m_Momentum) {
        super(m_MaxIteration, m_Neuron, m_LearningRate, m_Momentum);
    }

//    public DeltaRuleBatch(int m_MaxIteration, double m_LearningRate, 
//            double m_Momentum, double biasWeight, List<Double> weights) {
//        super(m_MaxIteration, new Neuron("no", biasWeight, weights), m_LearningRate, m_Momentum);
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
        // 1 Epoch
        List<List<Double>> sumDeltaWeights = new ArrayList<>();
        List<Double> sumDeltaBiasWeight = new ArrayList<>();

        for (int i = 0; i < instances.numInstances(); ++i) {
            List<List<Double>> deltaWeights = super.calculateDeltaWeight(i, deltaWeightPrev);
            List<Double> deltaBiasWeight = super.calculateDeltaBiasWeight(i, deltaBiasWeightPrev);
            if (i == 0) {
                sumDeltaWeights = deltaWeights;
                sumDeltaBiasWeight = deltaBiasWeight;
            } else {
                for (int j = 0; j < sumDeltaWeights.size(); ++j) {
                    for (int k = 0; k < sumDeltaWeights.get(j).size(); ++k) {
                        sumDeltaWeights.get(j).set(k, sumDeltaWeights.get(j).get(k) + deltaWeights.get(j).get(k));
                    }
                    sumDeltaBiasWeight.set(j, sumDeltaBiasWeight.get(j) + deltaBiasWeight.get(j));
                }
            }
        }

        for (int i = 0; i < getNeuron().size(); ++i) {
            for (int j = 0; j < getNeuron().get(i).getWeight().size(); ++j) {
                sumDeltaWeights.get(i).set(j, sumDeltaWeights.get(i).get(j) + (getMomentum() * deltaWeightPrev.get(i).get(j)));
            }
            sumDeltaBiasWeight.set(i, sumDeltaBiasWeight.get(i) + (getMomentum() * deltaBiasWeightPrev.get(i)));
        }
        deltaWeightPrev = sumDeltaWeights;
        deltaBiasWeightPrev = sumDeltaBiasWeight;

        super.updateWeights(sumDeltaWeights, sumDeltaBiasWeight);
    }

    private void print1Epoch(int epoch) {
        System.out.println("==================================");
        System.out.println("Model Pembelajaran Epoch " + epoch + "\n");
        super.printModel();
        System.out.println("==================================\n");
    }

    /**
     *
     * @param instances  data train
     * @throws Exception if classifier failed to build
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        super.setInstances(instances);
        double mse = 1.0;
        int iteration = 1;

        deltaWeightInitiation();
        if (super.getMaxIteration() == 0) {
            while (Double.compare(mse, THRESHOLD) > 0 && Double.compare(mse, Double.POSITIVE_INFINITY) < 0) {
                learning(instances);
                print1Epoch(iteration);
                mse = super.calculateMSE();
                System.out.println("MSE: " + mse);
                ++iteration;
            }
        } else { // stop if convergen before reaching max iteration
            while (iteration <= super.getMaxIteration() && Double.compare(mse, THRESHOLD) > 0
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
