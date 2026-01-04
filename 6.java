import java.util.*;

class NaiveBayes {

    private double[][] means;
    private double[][] variances;
    private double[] priors;
    private int[] classes;

    // Constructor (training happens here)
    public NaiveBayes(double[][] X, int[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        // Find unique classes
        Set<Integer> classSet = new HashSet<>();
        for (int label : y) {
            classSet.add(label);
        }

        classes = classSet.stream().mapToInt(Integer::intValue).toArray();
        int nClasses = classes.length;

        means = new double[nClasses][nFeatures];
        variances = new double[nClasses][nFeatures];
        priors = new double[nClasses];

        // Compute mean, variance, and prior for each class
        for (int c = 0; c < nClasses; c++) {
            int currentClass = classes[c];

            List<double[]> classSamples = new ArrayList<>();
            for (int i = 0; i < nSamples; i++) {
                if (y[i] == currentClass) {
                    classSamples.add(X[i]);
                }
            }

            priors[c] = (double) classSamples.size() / nSamples;

            for (int j = 0; j < nFeatures; j++) {
                double sum = 0.0;
                for (double[] sample : classSamples) {
                    sum += sample[j];
                }
                means[c][j] = sum / classSamples.size();

                double varSum = 0.0;
                for (double[] sample : classSamples) {
                    varSum += Math.pow(sample[j] - means[c][j], 2);
                }
                variances[c][j] = varSum / classSamples.size() + 1e-9; // stability
            }
        }
    }

    // Gaussian Probability Density Function
    private double gaussianPDF(double x, double mean, double var) {
        double numerator = Math.exp(-Math.pow(x - mean, 2) / (2 * var));
        double denominator = Math.sqrt(2 * Math.PI * var);
        return numerator / denominator;
    }

    // Predict one sample
    public int predictOne(double[] x) {
        double maxPosterior = Double.NEGATIVE_INFINITY;
        int bestClass = -1;

        for (int c = 0; c < classes.length; c++) {
            double logPrior = Math.log(priors[c]);
            double logLikelihood = 0.0;

            for (int j = 0; j < x.length; j++) {
                logLikelihood += Math.log(
                        gaussianPDF(x[j], means[c][j], variances[c][j])
                );
            }

            double posterior = logPrior + logLikelihood;

            if (posterior > maxPosterior) {
                maxPosterior = posterior;
                bestClass = classes[c];
            }
        }

        return bestClass;
    }

    // Predict multiple samples
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictOne(X[i]);
        }
        return predictions;
    }
}
