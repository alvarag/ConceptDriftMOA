/*
 * IB3.java
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
import java.util.Random;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.classifiers.lazy.neighboursearch.NormalizableDistance;
import moa.core.Measurement;

/**
 * Implementation of Instance-Based Learning (IB3). Presented in:
 * Aha, D. W., Kibler, D., & Albert, M. K. (1991). 
 * Instance-based learning algorithms. 
 * Machine learning, 6(1), 37-66.<br>
 * 
 * Fragments of code from: https://github.com/mblachnik/infoSel/
 * 
 * <p>
 * Valid options are:
 * <p>
 * -a confidence factor for acceptance <br>
 * -r confidence factor for removal <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160518
 */
public class IB3 extends AbstractClassifier {

	private static final long serialVersionUID = 6741600855950416603L;

	public FloatOption mAcceptable = new FloatOption("acceptable", 'a',
			"Confidence factor for accept", 0.90, 0, 1);

	public FloatOption mRemovable = new FloatOption("removable", 'r',
			"Confidence factor for remove", 0.75, 0, 1);

	private Instances mWindow;

	private NearestNeighbourSearch mSearch;

	private ArrayList<int[]> mClassRecord;

	private int mClasses;

	private int[] mFreqClasses;
	
	private Random mRandomGen;

	@Override
	public boolean isRandomizable() {
		
		return true;
	}

	@Override
    public double[] getVotesForInstance(Instance inst) {
		double v[] = new double[mClasses];
		
		try {
			if (mWindow.numInstances() > 0) {	
				Instance neighbour = mSearch.nearestNeighbour(inst);
				
				v[(int)neighbour.classValue()]++;
			}
		} catch(Exception e) {
			System.err.println("Error: kNN search failed.");
			e.printStackTrace();
			return new double[inst.numClasses()];
		}
		
		return v;
    }
	
	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			mWindow = new Instances(context, 0);
			mWindow.setClassIndex(context.classIndex());
			mClasses = mWindow.numClasses();
		} catch (Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	public void resetLearningImpl() {
		mWindow = null;
		mFreqClasses = null;
		mSearch = null;
		mClassRecord = null;
		mRandomGen = null;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		ArrayList<Integer> indexToRemove = new ArrayList<>();
		double dis[] = new double[mWindow.numInstances()];
		Instance sel = null;
		double bestDis = Double.POSITIVE_INFINITY;
		boolean aceptable = false;
		int indexSel;

		if (mRandomGen == null) {
			mRandomGen = new Random(randomSeed);
		}
		
		if (mFreqClasses == null) {
			mFreqClasses = new int[mClasses];
		}
		
		if (mWindow == null) {
			mWindow = new Instances(inst.dataset());
		}
		
		if (mSearch == null) {
			mSearch = new LinearNNSearch(mWindow);
			((NormalizableDistance)mSearch.getDistanceFunction()).setDontNormalize(true);
		}
		
		if (mClassRecord == null) {
			mClassRecord = new ArrayList<int[]>();
		}
		
		if (mWindow.numInstances() == 0) {
			mWindow.add(inst);
			mClassRecord.add(new int[]{1, 0});
			mFreqClasses[(int) inst.classValue()]++;
			return;
		}
		
		// Step 1 of the IB3
		// Compute all distances
		for (int j = 0; j < mWindow.numInstances(); j++)
			dis[j] = mSearch.getDistanceFunction().distance(mWindow.instance(j), inst);

		// Step 2 of the IB3
		// Look for the nearest "acceptable"
		for (int j = 0; j < mWindow.numInstances(); j++) {
			if (isAcceptable(mWindow.instance(j), j)) {
				if (dis[j] < bestDis) {
					aceptable = true;
					indexSel = j;
					bestDis = dis[j];
					sel = mWindow.instance(j);
				}
			}
		}

		// If no one is acceptable, take randomly one
		if (!aceptable) {
			indexSel = mRandomGen.nextInt(mWindow.numInstances());
			sel = mWindow.instance(indexSel);
		}

		// Step 3 of the IB3
		// If the class predicted is not correct -> Add inst as a new concept
		if (inst.classValue() != sel.classValue()) {
			mWindow.add(inst);
			mClassRecord.add(new int[]{1, 0});
			mFreqClasses[(int) inst.classValue()]++;
			
			try {
				mSearch.update(inst);
			} catch (Exception e) {
				System.err.println("Error: kNN update failed.");
				e.printStackTrace();
			}
		}

		// Step 4 of the IB3
		// Update classification record
		for (int i = 0; i < dis.length; i++) {
			if (dis[i] <= bestDis) {
				if (mWindow.instance(i).classValue() == inst.classValue()) {
					mClassRecord.get(i)[0]++;
				}
				else {
					mClassRecord.get(i)[1]++;
				}
				
				if (isRemovable(mWindow.instance(i), i)) {
					indexToRemove.add(i);
				}
			}
		}
		
		// Remove instances marked
		if (indexToRemove.size() > 0) {
			// Remove
			for (int i = indexToRemove.size() - 1; i >= 0; i--) {
				mFreqClasses[(int)mWindow.instance(i).classValue()]--;
				mWindow.delete(i);
				mClassRecord.remove(i);
			}
			
			try {
				mSearch.setInstances(mWindow);
			} catch (Exception e) {
				System.err.println("Error: kNN set failed.");
				e.printStackTrace();
			} 
		}
	}

	/**
	 * Computes if the instance is acceptable.
	 * 
	 * @param inst Instance.
	 * @param instIndex Index of the instance in the window.
	 * @return True if inst is acceptable.
	 */
	private boolean isAcceptable(Instance inst, int instIndex) {
		double n, aux;
		double minInst, minClass;
		
		n = (double) (mClassRecord.get(instIndex)[0] + mClassRecord.get(instIndex)[1]);
		aux = (double) mClassRecord.get(instIndex)[0];

		minInst = minConfidence(aux, n, mAcceptable.getValue());

		n = 0;
		for (int i = 0; i < mClasses; i++)
			n = n + (double) mFreqClasses[i];

		aux = (double) mFreqClasses[(int)inst.classValue()];

		minClass = minConfidence(aux, n, mAcceptable.getValue());

		if (minInst > minClass)
			return true;

		return false;
	}

	/**
	 * Computes if an instance is removable.
	 * 
	 * @param inst Instance.
	 * @param instIndex Index of the instance in the window.
	 * @return True if inst is removable.
	 */
	boolean isRemovable(Instance inst, int instIndex) {
		double n, aux;
		double maxInst, maxClass;
		
		n = (double) (mClassRecord.get(instIndex)[0] + mClassRecord.get(instIndex)[1]);
		aux = (double) mClassRecord.get(instIndex)[0];

		maxInst = maxConfidence(aux, n, mRemovable.getValue());

		n = 0;
		for (int i = 0; i < mClasses; i++)
			n = n + (double) mFreqClasses[i];

		aux = (double) mFreqClasses[(int)inst.classValue()];

		maxClass = maxConfidence(aux, n, mRemovable.getValue());

		if (maxInst < maxClass)
			return true;

		return false;
	}
	
    /**
     * Returns high end of confidence interval, given:
     *      
     * @param y the number of successes
     * @param n the number of trials
     * @param z the confidence level
     * @return max confidence.
     */
    private double maxConfidence(double y, double n, double z) {
        if (n == 0.0) {
            return 1;
        } else {
            double frequency = y / n;
            double z2 = z * z;
            double n2 = n * n;
            double numerator, denominator, val;
            val = z * Math.sqrt((frequency * (1.0 - frequency) / n) + z2 / (4 * n2));
            numerator = frequency + z2 / (2 * n) + val;
            denominator = 1.0 + z2 / n;
            return (numerator / denominator);
        }
    }

    /**
     * Returns low end of confidence interval, given:
     *      
     * @param y the number of successes
     * @param n the number of trials
     * @param z the confidence level
     * @return min confidence.
     */
    private double minConfidence(double y, double n, double z) {        
        if (n == 0.0) {
            return 0;
        } else {
            double frequency = y / n;
            double z2 = z * z;
            double n2 = n * n;
            double numerator, denominator, val;
            val = z * Math.sqrt((frequency * (1.0 - frequency) / n) + z2 / (4 * n2));
            numerator = frequency + z2 / (2 * n) - val;
            denominator = 1.0 + z2 / n;
            return (numerator / denominator);
        }
    }
	
	@Override
	public String getPurposeString() {

		return "IB3.";
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {

		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

}
