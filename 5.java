class LogisticRegression {

    private double[] weights;
    private double bias;

    // Constructor
    public LogisticRegression() {
        this.bias = 0.0;
    }

    // Sigmoid function
    private double sigmoid(double z) {
        // Prevent overflow
        if (z < -500) z = -500;
        if (z > 500) z = 500;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // Train the model
    public void fit(double[][] X, double[] y, int numIterations, double learningRate) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        weights = new double[nFeatures];

        for (int iter = 0; iter < numIterations; iter++) {

            double[] dw = new double[nFeatures];
            double db = 0.0;

            for (int i = 0; i < nSamples; i++) {
                double z = bias;
                for (int j = 0; j < nFeatures; j++) {
                    z += X[i][j] * weights[j];
                }

                double h = sigmoid(z);
                double error = h - y[i];

                for (int j = 0; j < nFeatures; j++) {
                    dw[j] += X[i][j] * error;
                }
                db += error;
            }

            // Update parameters
            for (int j = 0; j < nFeatures; j++) {
                weights[j] -= learningRate * (dw[j] / nSamples);
            }
            bias -= learningRate * (db / nSamples);
        }
    }

    // Predict probabilities
    public double predictProbability(double[] x) {
        double z = bias;
        for (int j = 0; j < x.length; j++) {
            z += x[j] * weights[j];
        }
        return sigmoid(z);
    }

    // Predict class (0 or 1)
    public int predict(double[] x) {
        return predictProbability(x) >= 0.5 ? 1 : 0;
    }
}
