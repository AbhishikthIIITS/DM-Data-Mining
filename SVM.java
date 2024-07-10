public class SVM {
    private static double[][] data = {
            {4, 1}, {4, -2}, {5, -2}, {6, 3},  // S1
            {1, 2}, {2, -1}, {1, -2}, {-1, 1}   // S2
    };

    private static double[] labels = {1, 1, 1, 1, -1, -1, -1, -1};

    private static double[] alpha = new double[data.length];
    private static double bias = 0;
    private static double[] weights;

    public static void train() {
        int numSamples = data.length;

        // SMO algorithm
        boolean hasConverged;
        do {
            int numChangedAlphas = 0;
            for (int i = 0; i < numSamples; i++) {
                double error_i = calculateError(i);

                if ((labels[i] * error_i < -0.001 && alpha[i] < 1) || (labels[i] * error_i > 0.001 && alpha[i] > 0)) {
                    int j = selectSecondAlpha(i);
                    double error_j = calculateError(j);

                    double alpha_i_old = alpha[i];
                    double alpha_j_old = alpha[j];

                    // Update alpha values
                    updateAlpha(i, j, error_i, error_j);

                    // Update bias
                    updateBias(i, j, alpha_i_old, alpha_j_old, error_i, error_j);

                    numChangedAlphas++;
                }
            }

            if (numChangedAlphas == 0) {
                hasConverged = true;
            } else {
                hasConverged = false;
            }
        } while (!hasConverged);

        // Calculate weights
        calculateWeights();
    }

    private static double calculateError(int i) {
        double prediction = bias;
        for (int j = 0; j < data.length; j++) {
            prediction += alpha[j] * labels[j] * linearKernel(data[i], data[j]);
        }
        return prediction - labels[i];
    }

    private static int selectSecondAlpha(int i) {
        // Select the second alpha that maximizes the error difference
        int j = -1;
        double error_i = calculateError(i);
        double maxErrorDiff = 0;

        for (int k = 0; k < alpha.length; k++) {
            double error_k = calculateError(k);
            double errorDiff = Math.abs(error_i - error_k);

            if (errorDiff > maxErrorDiff) {
                maxErrorDiff = errorDiff;
                j = k;
            }
        }

        return j;
    }

    private static void updateAlpha(int i, int j, double error_i, double error_j) {
        double eta = 2 * linearKernel(data[i], data[j]) - linearKernel(data[i], data[i]) - linearKernel(data[j], data[j]);

        if (eta >= 0) {
            return;
        }

        double alpha_i_old = alpha[i];
        double alpha_j_old = alpha[j];

        double alpha_j_new = alpha_j_old - labels[j] * (error_i - error_j) / eta;
        double L = 0;
        double H = 0;

        if (labels[i] != labels[j]) {
            L = Math.max(0, alpha_j_old - alpha_i_old);
            H = Math.min(1, 1 + alpha_j_old - alpha_i_old);
        } else {
            L = Math.max(0, alpha_j_old + alpha_i_old - 1);
            H = Math.min(1, alpha_j_old + alpha_i_old);
        }

        if (alpha_j_new > H) {
            alpha_j_new = H;
        } else if (alpha_j_new < L) {
            alpha_j_new = L;
        }

        alpha[i] = alpha_i_old + labels[i] * labels[j] * (alpha_j_old - alpha_j_new);
        alpha[j] = alpha_j_new;
    }

    private static void updateBias(int i, int j, double alpha_i_old, double alpha_j_old, double error_i, double error_j) {
        double b1 = bias - error_i - labels[i] * (alpha[i] - alpha_i_old) * linearKernel(data[i], data[i])
                - labels[j] * (alpha[j] - alpha_j_old) * linearKernel(data[i], data[j]);

        double b2 = bias - error_j - labels[i] * (alpha[i] - alpha_i_old) * linearKernel(data[i], data[j])
                - labels[j] * (alpha[j] - alpha_j_old) * linearKernel(data[j], data[j]);

        if (0 < alpha[i] && alpha[i] < 1) {
            bias = b1;
        } else if (0 < alpha[j] && alpha[j] < 1) {
            bias = b2;
        } else {
            bias = (b1 + b2) / 2;
        }
    }

    private static void calculateWeights() {
        int numFeatures = data[0].length;
        weights = new double[numFeatures];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                weights[j] += alpha[i] * labels[i] * data[i][j];
            }
        }
    }

    private static double linearKernel(double[] x1, double[] x2) {
        double sum = 0;
        for (int i = 0; i < x1.length; i++) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }

    public static void main(String[] args) {
       /* */
        train();

        // Display results
        System.out.println("Weight Vector:");
        for (double weight : weights) {
            System.out.print(weight + " ");
        }
        System.out.println("\nBias: " + bias);
    }
}
