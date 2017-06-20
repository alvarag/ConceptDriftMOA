package utils.stats;

import weka.core.Utils;

public class ChiSquare {

    public static double[][] testvalues = new double[][] {
            new double[] { Double.NaN, 0.05, 0.01, 0.001 },
            new double[] { 1, 3.84, 6.64, 10.83 },
            new double[] { 2, 5.99, 9.21, 13.82 },
            new double[] { 3, 7.82, 11.35, 16.27 },
            new double[] { 4, 9.49, 13.28, 18.47 },
            new double[] { 5, 11.07, 15.09, 20.52 },
            new double[] { 6, 12.59, 16.81, 22.46 },
            new double[] { 7, 14.07, 18.48, 24.32 },
            new double[] { 8, 15.51, 20.09, 26.12 },
            new double[] { 9, 16.92, 21.67, 27.88 },
            new double[] { 10, 18.31, 23.21, 29.59 },
            new double[] { 11, 19.68, 24.73, 31.26 },
            new double[] { 12, 21.03, 26.22, 32.91 },
            new double[] { 13, 22.36, 27.69, 34.53 },
            new double[] { 14, 23.69, 29.14, 36.12 },
            new double[] { 15, 25.00, 30.58, 37.70 },
            new double[] { 16, 26.30, 32.00, 39.25 },
            new double[] { 17, 27.59, 33.41, 40.79 },
            new double[] { 18, 28.87, 34.81, 42.31 },
            new double[] { 19, 30.14, 36.19, 43.82 },
            new double[] { 20, 31.41, 37.57, 45.32 },
            new double[] { 21, 32.67, 38.93, 46.80 },
            new double[] { 22, 33.92, 40.29, 48.27 },
            new double[] { 23, 35.17, 41.64, 49.73 },
            new double[] { 24, 36.42, 42.98, 51.18 },
            new double[] { 25, 37.65, 44.31, 52.62 },
            new double[] { 26, 38.89, 45.64, 54.05 },
            new double[] { 27, 40.11, 46.96, 55.48 },
            new double[] { 28, 41.34, 48.28, 56.89 },
            new double[] { 29, 42.56, 49.59, 58.30 },
            new double[] { 30, 43.77, 50.89, 59.70 },
            new double[] { 31, 44.99, 52.19, 61.10 },
            new double[] { 32, 46.19, 53.49, 62.49 },
            new double[] { 33, 47.40, 54.78, 63.87 },
            new double[] { 34, 48.60, 56.06, 65.25 },
            new double[] { 35, 49.80, 57.34, 66.62 },
            new double[] { 36, 51.00, 58.62, 67.99 },
            new double[] { 37, 52.19, 59.89, 69.35 },
            new double[] { 38, 53.38, 61.16, 70.70 },
            new double[] { 39, 54.57, 62.43, 72.06 },
            new double[] { 40, 55.76, 63.69, 73.40 },
            new double[] { 41, 56.94, 64.95, 74.75 },
            new double[] { 42, 58.12, 66.21, 76.08 },
            new double[] { 43, 59.30, 67.46, 77.42 },
            new double[] { 44, 60.48, 68.71, 78.75 },
            new double[] { 45, 61.66, 69.96, 80.08 },
            new double[] { 46, 62.83, 71.20, 81.40 },
            new double[] { 47, 64.00, 72.44, 82.72 },
            new double[] { 48, 65.17, 73.68, 84.04 },
            new double[] { 49, 66.34, 74.92, 85.35 },
            new double[] { 50, 67.51, 76.15, 86.66 },
            new double[] { 51, 68.67, 77.39, 87.97 },
            new double[] { 52, 69.83, 78.62, 89.27 },
            new double[] { 53, 70.99, 79.84, 90.57 },
            new double[] { 54, 72.15, 81.07, 91.87 },
            new double[] { 55, 73.31, 82.29, 93.17 },
            new double[] { 56, 74.47, 83.51, 94.46 },
            new double[] { 57, 75.62, 84.73, 95.75 },
            new double[] { 58, 76.78, 85.95, 97.04 },
            new double[] { 59, 77.93, 87.17, 98.32 },
            new double[] { 60, 79.08, 88.38, 99.61 },
            new double[] { 61, 80.23, 89.59, 100.89 },
            new double[] { 62, 81.38, 90.80, 102.17 },
            new double[] { 63, 82.53, 92.01, 103.44 },
            new double[] { 64, 83.68, 93.22, 104.72 },
            new double[] { 65, 84.82, 94.42, 105.99 },
            new double[] { 66, 85.97, 95.63, 107.26 },
            new double[] { 67, 87.11, 96.83, 108.53 },
            new double[] { 68, 88.25, 98.03, 109.79 },
            new double[] { 69, 89.39, 99.23, 111.06 },
            new double[] { 70, 90.53, 100.43, 112.32 },
            new double[] { 71, 91.67, 101.62, 113.58 },
            new double[] { 72, 92.81, 102.82, 114.84 },
            new double[] { 73, 93.95, 104.01, 116.09 },
            new double[] { 74, 95.08, 105.20, 117.35 },
            new double[] { 75, 96.22, 106.39, 118.60 },
            new double[] { 76, 97.35, 107.58, 119.85 },
            new double[] { 77, 98.48, 108.77, 121.10 },
            new double[] { 78, 99.62, 109.96, 122.35 },
            new double[] { 79, 100.75, 111.14, 123.59 },
            new double[] { 80, 101.88, 112.33, 124.84 },
            new double[] { 81, 103.01, 113.51, 126.08 },
            new double[] { 82, 104.14, 114.70, 127.32 },
            new double[] { 83, 105.27, 115.88, 128.57 },
            new double[] { 84, 106.40, 117.06, 129.80 },
            new double[] { 85, 107.52, 118.24, 131.04 },
            new double[] { 86, 108.65, 119.41, 132.28 },
            new double[] { 87, 109.77, 120.59, 133.51 },
            new double[] { 88, 110.90, 121.77, 134.75 },
            new double[] { 89, 112.02, 122.94, 135.98 },
            new double[] { 90, 113.15, 124.12, 137.21 },
            new double[] { 91, 114.27, 125.29, 138.44 },
            new double[] { 92, 115.39, 126.46, 139.67 },
            new double[] { 93, 116.51, 127.63, 140.89 },
            new double[] { 94, 117.63, 128.80, 142.12 },
            new double[] { 95, 118.75, 129.97, 143.34 },
            new double[] { 96, 119.87, 131.14, 144.57 },
            new double[] { 97, 120.99, 132.31, 145.79 },
            new double[] { 98, 122.11, 133.48, 147.01 },
            new double[] { 99, 123.23, 134.64, 148.23 },
            new double[] { 100, 124.34, 135.81, 149.45 } };
    
    static boolean chi2DistributionTest(double[] dist1, double[] dist2){
        if (dist1.length>2) {
            double[][] o=optimize(dist1, dist2);
            dist1=o[0];
            dist2=o[2];
        }
        int k=dist1.length;
        double sum1=Utils.sum(dist1);
        double sum2=Utils.sum(dist2);
        double chi2=0;
        for (int i=0;i<k;i++){
            chi2=chi2+dist1[i]*dist1[i]/((dist1[i]+dist2[i])*sum1);
            chi2=chi2+dist2[i]*dist2[i]/((dist1[i]+dist2[i])*sum2);
        }
        chi2=(sum1+sum2)*(chi2-1);
        return chi2<=testvalues[k-1][1];
    }

    private static double[][] optimize(double[] dist1, double[] dist2) {
        int[] sort1=Utils.sort(dist1);
        if (dist1[sort1[0]]<10){
            
        }
        int t=0;
        double sum=0;
        
        return null;
    }

}
