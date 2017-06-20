/*
 * SyncStream.java
 * Copyright (C) 2016 Burgos University, Spain 
 * @author Álvar Arnaiz-González
 *     
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *     
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *     
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package moa.classifiers.lazy;

import java.util.ArrayList;

import Jama.Matrix;
import VisualNumerics.math.Statistics;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

// From https://github.com/kramerlab/SyncStream/
import main.java.PCA;
import main.java.PTree;
import main.java.SyncObject;

/**
 * Wrapper for SyncStream - presented in: 
 * Shao, J., Ahmadi, Z., & Kramer, S. (2014, August). Prototype-based 
 * learning on concept-drifting data streams. In Proceedings of the 
 * 20th ACM SIGKDD international conference on Knowledge discovery 
 * and data mining (pp. 412-421). ACM.<br>
 * Code available at: https://github.com/kramerlab/SyncStream/
 * <p>
 * Valid options are:
 * <p>
 * -s strategy to use: PCA or statistic.<br>
 * -a strategy to use in PCA: average or maximum.<br> 
 * -t the theta angle to use.<br> 
 * -n the number of objects to calculate the concept change.<br> 
 *
 * @author Álvar Arnaiz-González (alvarag@ubu.es)
 * @version 20160427
 */
public class SyncStream extends AbstractClassifier {

	private static final long serialVersionUID = 7169001197365113653L;

	public MultiChoiceOption mStrategyOption = new MultiChoiceOption(
	        "strategy", 's', "Strategy to use", new String[] {"PCA","Statistic"}, 
	        new String[] { "PCA", "Statistic" }, 0);

	public MultiChoiceOption mAngle = new MultiChoiceOption(
	        "angle", 'a', "Strategy to use in PCA", new String[] {"AVG","Max."}, 
	        new String[] { "AVG", "Maximum" }, 0);

	public IntOption mThetaOption = new IntOption( "theta", 't', "The theta angle", 
	        60, 1, 360);
	
	public IntOption mNumObjectOption = new IntOption( "numObject", 'n', 
	        "Number of objects for concept change", 250, 1, Integer.MAX_VALUE);
	
	private PTree mPTree;

	private ArrayList<SyncObject> mPreData;
	
	private ArrayList<SyncObject> mNextData;
	
	private ArrayList<Double> mAngles;
	
	private int mNumClasses;

	private int mConceptID;
	
	private int mNumObj;

	@Override
	public boolean isRandomizable() {

		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		double[] pred = new double[inst.classAttribute().numValues()];
		int preLabel = -1;

		if (mPTree != null)
			preLabel = kNNPTreeClassifier(mPTree, convertInstToSync(inst));
		
		for (int i = 0; i < pred.length; i++)
			if (i == preLabel)
				pred[i] = 1;
			else
				pred[i] = 0;

		return pred;
	}

	private int kNNPTreeClassifier(PTree sht, SyncObject test) {
		int prelabel = 0;

		// check inputs for validity
		if (sht.prototypeLevel.size() == 0)
			return 0; // bad input

		ArrayList<SyncObject> data = new ArrayList<SyncObject>();

		// prototype objects
		for (int i = 0; i < sht.prototypeLevel.size(); i++) {
			data.add(sht.prototypeLevel.get(i));
		}

		// check inputs for validity
		if (data.size() == 0)
			return 0; // bad input

		double[] dis = new double[data.size()];
		for (int s = 0; s < data.size(); s++) {
			dis[s] = dist(data.get(s).data, test.data);
		}

		double minV = Double.MAX_VALUE;
		int idd = -1;
		for (int j = 0; j < data.size(); j++) {
			if (dis[j] < minV) {
				minV = dis[j];
				idd = j;
			}
		}

		if (idd == -1) {
			System.out.println("Error:" + dis[0]);
		}
		
		prelabel = data.get(idd).label;
		
		return prelabel;
	}

	public double EuclideanDist(double[] dis) {

		double val = 0.0;
		for (int i = 0; i < dis.length; i++) {
			val = val + dis[i] * dis[i];
		}
		double dist = Math.sqrt(val);
		
		return dist;
	}

	public double dist(double[] al1, double[] al2) {
		double res = 0.0;

		if (al1.length != al2.length)
			return 0.0;
		else {
			double[] diss = new double[al1.length];
			for (int d = 0; d < al1.length; d++) {
				diss[d] = al1[d] - al2[d];
			}
			res = EuclideanDist(diss);
		}

		return res;
	}

	@Override
	public void resetLearningImpl() {
		mPTree = new PTree();
		mPreData = new ArrayList<SyncObject>();
		mNextData = new ArrayList<SyncObject>();
		mAngles = new ArrayList<Double>();
		mNumClasses = 0;
		mConceptID = 0;
		mNumObj = 0;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		SyncObject sync = convertInstToSync(inst);
		
		mPTree.addInstanceToTree(sync);

		sync.conceptType = mConceptID;
		mNextData.add(sync);
		
		mNumObj++;

		if (mNumObj == mNumObjectOption.getValue())
			checkConcept ();
	}
	
	private void checkConcept () {
		// PCA
		if (mStrategyOption.getChosenIndex() == 0) {
			/* Second Strategy: PCA Analysis */
			double orient = -1.0;
			if (mPreData.size() > 0 && mNextData.size() > 0) {
				orient = 180 * PCA_Angle(mPreData, mNextData, mNumClasses + 1)
				          / Math.PI;
				mAngles.add(orient);
			}

			if (mPreData.size() > 0 && orient > mThetaOption.getValue()) {
				mPTree.addConcept(mConceptID, true);
				mConceptID++;
			}
			mPreData = new ArrayList<SyncObject>();
			
			for (int m = 0; m < mNextData.size(); m++)
				mPreData.add(mNextData.get(m));
		}
		// Statistic
		else if (mStrategyOption.getChosenIndex() == 1) {
			/* Third Strategy: PCA Analysis */
			double pvalue = 1.0;
			if (mPreData.size() > 0 && mNextData.size() > 0) {
				pvalue = computePvalue(mPreData, mNextData, mNumClasses);
				mAngles.add(pvalue);
			}

			if (mPreData.size() > 0 && pvalue < 0.01) {
				mPTree.addConcept(mConceptID, true);
				mConceptID++;
			}
			mPreData = new ArrayList<SyncObject>();
			for (int m = 0; m < mNextData.size(); m++) {
				mPreData.add(mNextData.get(m));
			}
		}

		mNumObj = 0;
		mNextData = new ArrayList<SyncObject>();
	}

	private double PCA_Angle(ArrayList<SyncObject> a, ArrayList<SyncObject> b, int C) {
		double maxT = 0.0;
		double avgT = 0.0;
		int n = 0;

		for (int k = 0; k < C; k++) {
			int n1 = 0, n2 = 0;
			double dp = 0.0, theta = 0.0;
			for (int i = 0; i < a.size(); i++) {
				if (a.get(i).label == k) {
					n1++;
				}
			}

			for (int i = 0; i < b.size(); i++) {
				if (b.get(i).label == k) {
					n2++;
				}
			}
			if (n1 > 20 && n2 > 20) { // at least some objects are existed
				double[][] temp1 = new double[n1][];
				double[][] temp2 = new double[n2][];
				int t1 = 0, t2 = 0;
				for (int i = 0; i < a.size(); i++) {
					if (a.get(i).label == k) {
						temp1[t1] = a.get(i).data;
						t1++;
					}
				}
				for (int i = 0; i < b.size(); i++) {
					if (b.get(i).label == k) {
						temp2[t2] = b.get(i).data;
						t2++;
					}
				}

				Matrix m1 = new Matrix(temp1);
				Matrix m2 = new Matrix(temp2);
				m1 = m1.transpose();
				m2 = m2.transpose();

				PCA pca1 = new PCA();
				boolean f1 = pca1.eigenPCA(m1, false, false);
				
				PCA pca2 = new PCA();
				boolean f2 = pca2.eigenPCA(m2, false, false);
				if (f1 && f2) {

					double[] fC1 = pca1.getFirstPC();
					double[] fC2 = pca2.getFirstPC();

					for (int i = 0; i < fC1.length; i++) {
						dp = dp + fC1[i] * fC2[i];
					}

					if (dp != 0) {
						theta = Math.acos(Math.abs(dp));
						if (theta > Math.PI / 2)
							theta = Math.PI - theta;
						avgT += theta;
						n++;
						if (theta > maxT) {
							maxT = theta;
						}
					}
				}
			}
		}

//		if (angleC == "avg")
		if (mAngle.getChosenIndex() == 0)
			return maxT;
		else if (n > 0)
			return avgT / n;
		else
			return 0.0;
	}
	
	public double computePvalue(ArrayList<SyncObject> predata,
	               ArrayList<SyncObject> nextdata, int numC) {
		double pvalue = 1.0;
		int nsize = predata.size();
		int dim = predata.get(0).data.length;

		for (int i = 0; i < numC; i++) {
			ArrayList<SyncObject> dx = new ArrayList<SyncObject>();
			ArrayList<SyncObject> dy = new ArrayList<SyncObject>();
			for (int j = 0; j < nsize; j++) {
				if (predata.get(j).label == (i + 1)) {
					dx.add(predata.get(j));
				}
				if (nextdata.get(j).label == (i + 1)) {
					dy.add(nextdata.get(j));
				}
			}

			if (dx.size() > 20 && dy.size() > 20) {
				double S1 = 0.0, S2 = 0.0;
				double R1 = 0.0, R2 = 0.0;
				int lenx = dx.size();
				int leny = dy.size();

				for (int d = 0; d < dim; d++) {
					double[][] X = new double[lenx][2];
					double[][] Y = new double[leny][2];
					double[][] Z = new double[lenx + leny][2];

					double[][] XX = new double[lenx][2];
					double[][] YY = new double[leny][2];
					double[][] ZZ = new double[lenx + leny][2];

					for (int k = 0; k < lenx; k++) {
						X[k][0] = ((SyncObject) dx.get(k)).data[d];
						X[k][1] = k + 1;
						Z[k][0] = X[k][0];
						Z[k][1] = k + 1;
					}

					for (int k = 0; k < leny; k++) {
						Y[k][0] = ((SyncObject) dy.get(k)).data[d];
						Y[k][1] = k + 1;
						Z[k + lenx][0] = Y[k][0];
						Z[k + lenx][1] = k + 1 + lenx;
					}

					quickSort(X);
					quickSort(Y);
					quickSort(Z);

					for (int k = 0; k < lenx; k++) {
						XX[k][0] = X[k][1];
						XX[k][1] = k + 1;
						ZZ[k][0] = Z[k][1];
						ZZ[k][1] = k + 1;
					}

					for (int k = 0; k < leny; k++) {
						YY[k][0] = Y[k][1];
						YY[k][1] = k + 1;
						ZZ[k + lenx][0] = Z[k + lenx][1];
						ZZ[k + lenx][1] = k + 1 + lenx;
					}

					quickSort(XX);
					quickSort(YY);
					quickSort(ZZ);

					double r1 = 0, r2 = 0;

					for (int k = 0; k < lenx; k++) {
						r1 = r1 + ZZ[k][1];
					}
					for (int k = 0; k < leny; k++) {
						r2 = r2 + ZZ[k + lenx][1];
					}

					R1 = R1 + r1;
					R2 = R2 + r2;

					r1 = r1 / lenx;
					r2 = r2 / leny;

					double s1 = 0, s2 = 0;

					for (int k = 0; k < lenx; k++) {
						s1 = s1
								+ (ZZ[k][1] - XX[k][1] - r1 + (lenx + 1.0) / 2.0)
								* (ZZ[k][1] - XX[k][1] - r1 + (lenx + 1.0) / 2.0);
					}

					for (int k = 0; k < leny; k++) {
						s2 = s2
								+ (ZZ[k + lenx][1] - YY[k][1] - r2 + (leny + 1.0) / 2.0)
								* (ZZ[k + lenx][1] - YY[k][1] - r2 + (leny + 1.0) / 2.0);
					}

					S1 = S1 + s1;
					S2 = S2 + s2;
				}

				S1 = S1 / ((lenx - 1) * dim);
				S2 = S2 / ((leny - 1) * dim);

				R1 = R1 / ((lenx - 1) * dim);
				R2 = R2 / ((leny - 1) * dim);

				double theta = Math.sqrt((lenx + leny) * S1 / leny
						+ (lenx + leny) * S2 / lenx);

				if (theta <= 1e-6) {
					theta = Math.sqrt((lenx + leny) / (2.0 * lenx * leny));
				}

				double W_BF = ((R1 - R2) / theta)
				       * Math.sqrt((lenx * leny) / ((lenx + leny) * dim)); // ??

				// sort the two data
				double p = 2.0 * Statistics.tCdf(-Math.abs(W_BF), lenx + leny - 1);
				if (p < pvalue)
					pvalue = p;
			}

		}

		return pvalue;
	}

	private void quickSort(double[][] array) {
		qsort(array, 0, array.length - 1);
	}

	private static void qsort(double[][] array, int le, int ri) {
		int lo = le, hi = ri;

		if (hi > lo) {
			// Pivotelement bestimmen
			double mid = array[(lo + hi) / 2][0];
			while (lo <= hi) {
				// Erstes Element suchen, das gr��er oder gleich dem
				// Pivotelement ist, beginnend vom linken Index
				while (lo < ri && array[lo][0] < mid)
					++lo;

				// Element suchen, das kleiner oder gleich dem
				// Pivotelement ist, beginnend vom rechten Index
				while (hi > le && array[hi][0] > mid)
					--hi;

				// Wenn Indexe nicht gekreuzt --> Inhalte vertauschen
				if (lo <= hi) {
					swap(array, lo, hi);
					++lo;
					--hi;
				}
			}
			// Linke Partition sortieren
			if (le < hi) {
				qsort(array, le, hi);
			}

			// Rechte Partition sortieren
			if (lo < ri) {
				qsort(array, lo, ri);
			}
		}
	}
	
	private static void swap(double[][] array, int idx1, int idx2) {
		double tmp1 = array[idx1][0];
		double tmp2 = array[idx1][1];
		array[idx1][0] = array[idx2][0];
		array[idx1][1] = array[idx2][1];
		array[idx2][0] = tmp1;
		array[idx2][1] = tmp2;
	}

	/**
	 * Transforms an Instance into SyncObject.
	 * 
	 * @param inst Instance to transform.
	 * @return SyncObject with the information of inst.
	 */
	private SyncObject convertInstToSync(Instance inst) {
		SyncObject sync;
		double[] data = new double[inst.numAttributes() - 1];

		for (int i = 0; i < inst.numAttributes() - 1; i++)
			if (inst.classIndex() != i)
				data[i] = inst.value(i);
			else
				i--;

		sync = new SyncObject(data, (int) inst.classValue());

		return sync;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		
	}

	@Override
	public String getPurposeString() {
		return "SyncStream";
	}

	@Override
	public void setModelContext(InstancesHeader context) {
		mNumClasses = context.numClasses();
	}
}
