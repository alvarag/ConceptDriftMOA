/*
 * LWF.java
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

import java.util.Hashtable;

import weka.core.Utils;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
 
/**
 * Implementation of Locally-Weighted Forgetting (LWF). Presented in:
 * Salganicoff, M. (1993, December). Density-adaptive learning and forgetting.
 * In Proceedings of the Tenth International Conference on Machine Learning
 * (Vol. 3, pp. 276-283).
 * <p>
 * Valid options are:
 * <p>
 * -f forgetting rate <br>
 * -t threshold <br>
 * -m maximum decay rate <br>
 * -k number of neighbors <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160509
 */
public class LWF extends WFkNN {

	private static final long serialVersionUID = 5470746585240664971L;

	public FloatOption mTheta = new FloatOption("theta", 't', 
	        "Theta", 0.9, 0, 1);

	public FloatOption mBeta = new FloatOption("beta", 'b', 
	        "Beta", 0.04, 0, 1);

	public FloatOption mMaximumDecayRate = new FloatOption("maxDecayRate", 'm',
	        "Maximum decay rate", 0.8, 0, 1);

	private Hashtable<String, Double> mWeightes;

	private NearestNeighbourSearch mSearch;

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		Instances nn;
		double[] distNN;
		double forgetting, w;
		boolean change = false;
		int k;

		if (mWindow == null)
			mWindow = new Instances(inst.dataset());

		if (mWeightes == null)
			mWeightes = new Hashtable<>();

		if (mSearch == null) {
			if (nearestNeighbourSearchOption.getChosenIndex()== 0) {
				mSearch = new LinearNNSearch(mWindow);
			} else {
				mSearch = new KDTree(mWindow);
			}
		}

		try {
			if (!mWeightes.containsKey(inst.toString())) {
				mWindow.add(inst);
				mWeightes.put(inst.toString(), 1.0);
				mSearch.update(inst);
			}

			// k + 2 because the first is always inst and the last only is
			// considered for the formule.
			k = (int)Math.ceil(mWindow.numInstances() * mBeta.getValue());
			nn = mSearch.kNearestNeighbours(inst, k + 2);
			distNN = mSearch.getDistances();

			// for each neighbor reduce its weight
			for (int i = 1; i < nn.numInstances() - 1; i++) {
				w = mWeightes.get(nn.get(i).toString());
				forgetting = mMaximumDecayRate.getValue() + 
				              ((1 - mMaximumDecayRate.getValue()) * 
				               (Math.pow(distNN[i], 2) / 
				                 Math.pow(distNN[distNN.length - 1], 2)));
				
				w *= forgetting;
				
				mWeightes.put(nn.get(i).toString(), w);
				
				if (w < mTheta.getValue()) {
					removeFromWindow(nn.get(i));
					change = true;
				}
			}
			
			// update the search if at least one instance is removed
			if (change)
				mSearch.setInstances(mWindow);
		} catch (Exception e) {
			 System.err.println("Error: kNN search failed.");
			 e.printStackTrace();
		}
	}

	/**
	 * Remove an instance from the window.
	 * 
	 * @param inst instance to remove from the window.
	 */
	private void removeFromWindow(Instance inst) {
		for (int i = 0; i < mWindow.numInstances(); i++) {
			if (compare(inst, mWindow.instance(i)) == 0) {
				mWeightes.remove(mWindow.instance(i).toString());
				mWindow.delete(i);
				return;
			}
		}
	}
	
	/**
	 * compares the two instances, returns -1 if o1 is smaller than o2, 0 if
	 * equal and +1 if greater. The method assumes that both instance objects
	 * have the same attributes, they don't have to belong to the same dataset.
	 * 
	 * @param inst1 the first instance to compare
	 * @param inst2 the second instance to compare
	 * @return returns -1 if inst1 is smaller than inst2, 0 if equal and +1 if
	 *         greater
	 */
	public static int compare(Instance inst1, Instance inst2) {
		int result = 0;

		for (int i = 0; i < inst1.numAttributes(); i++) {
			// comparing attribute values
			// 1. special handling if missing value (NaN) is involved:
			if (inst1.isMissing(i) || inst2.isMissing(i)) {
				if (inst1.isMissing(i) && inst2.isMissing(i)) {
					continue;
				} else {
					if (inst1.isMissing(i))
						result = -1;
					else
						result = 1;
					break;
				}
			}
			// 2. regular values:
			else {
				if (Utils.eq(inst1.value(i), inst2.value(i))) {
					continue;
				} else {
					if (inst1.value(i) < inst2.value(i))
						result = -1;
					else
						result = 1;
				}
			}
			if (result != 0)
				break;
		}

		return result;
	}
	
	@Override
	public void resetLearningImpl() {
		mWeightes = null;
		mWindow = null;
		mSearch = null;
	}

	public String getPurposeString() {
		return "Locally-Weighted Forgetting (LWF).";
	}

	@Override
	public void setModelContext(InstancesHeader context) {
		mWindow = new Instances(context, 0);
		mWindow.setClassIndex(context.classIndex());
		mNumClasses = context.classAttribute().numValues();
	}
}
