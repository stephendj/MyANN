package myann;

import java.util.List;
import weka.core.Instance;
import weka.core.Instances;

public class PerceptronTrainingRule extends SingleLayerPerceptron {

    private static final double THRESHOLD = 0.01;

    public PerceptronTrainingRule(int m_MaxIteration, Neuron m_Neuron,
            double m_LearningRate, double m_Momentum) {
        super(m_MaxIteration, m_Neuron, m_LearningRate, m_Momentum);
    }

    private void learning(Instances instances) {
        // 1 Epoch
        for (int i = 0; i < instances.numInstances(); ++i) {
            double output = super.calculateOutput(instances.instance(i));
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
        double error = 1.0;
        int iteration = 1;
        if (super.getMaxIteration() == 0) {
            while (Double.compare(error, THRESHOLD) > 0) {
                learning(instances);
                print1Epoch(iteration);
                error = super.calculateMSE();
                ++iteration;
            }
        } else {
            while (iteration <= super.getMaxIteration() && Double.compare(error, THRESHOLD) > 0) {
                learning(instances);
                print1Epoch(iteration);
                error = super.calculateMSE();
                ++iteration;
            }
        }
    }

}
