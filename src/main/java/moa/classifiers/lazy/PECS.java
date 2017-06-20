/*
 * PECS.java
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
import java.util.Hashtable;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
 
/**
 * Implementation of Prediction Error Context Switching (PECS). Presented in:
 * Salganicoff, M. (1997). Tolerating concept and sampling shift in lazy 
 * learning using prediction error context switching. 
 * Artificial Intelligence Review, 11(1-5), 133-155.
 * <p>
 * Valid options are:
 * <p>
 * -t inactivation threshold <br>
 * -m maximum decay rate <br>
 * -b beta value for compute the number of neighbors <br>
 * -r shift register length <br>
 * -c confidence value for the confidence interval <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160609
 */
public class PECS extends WFkNN {

	private static final long serialVersionUID = -5984959493215340279L;

	/**
	 * Inactivation threshold: p_max.
	 */
	public FloatOption mInactivationThreshold = new FloatOption("pMax", 'x', 
	        "Inactivation threshold", 0.4, 0, 1);

	/**
	 * Agreement acceptance probability: p_min.
	 */
	public FloatOption mAgreementAcceptance = new FloatOption("pMin", 'm',
	        "Agreement acceptance probability", 0.6, 0, 1);

	/**
	 * Used to compute the nearest neighbour number.
	 */
	public FloatOption mBeta = new FloatOption("beta", 'b',
	        "Beta for compute the number of neighbors", 0.04, 0, 1);

	/**
	 * Length of the agreement values register. 
	 */
	public IntOption mRegisterLenght = new IntOption("sizeSR", 'r',
	        "Shift register lenght", 20, 1, Integer.MAX_VALUE);

	/**
	 * Confidence value to use for the probability of agreement.
	 */
	public MultiChoiceOption mConfidenceValue = new MultiChoiceOption(
	        "confidenceValue", 'c', "Confidence value for the confidence intervals.",
	        new String[]{"99%", "95%", "90%", "80%"},
	        new String[]{"2.575829", "1.959964", "1.644854", "1.281552"}, 3);

	public static final double PERC_99 = 2.575829;
	
	public static final double PERC_95 = 1.959964;
	
	public static final double PERC_90 = 1.644854;
	
	public static final double PERC_80 = 1.281552;

	/**
	 * Hash of the instances in L.
	 */
	private Hashtable<String, Boolean> mHashInstancesL;

	/**
	 * Shift register for each instance.
	 */
	private Hashtable<String, List<Boolean>> mShiftRegisters;

	/**
	 * All instances: L u U. 
	 */
	private Instances mInstancesLU;
	
	/**
	 * Search over window.
	 */
	private NearestNeighbourSearch mWindowSearch;
	
	/**
	 * Search over all instances: L u U.
	 */
	private NearestNeighbourSearch mLUSearch;

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		List<Instance> nn = null;
		List<Boolean> shiftReg;
		double[] distNN;
		double x;
		int k;

		if (mWindow == null)
			mWindow = new Instances(inst.dataset());

		if (mInstancesLU == null)
			mInstancesLU = new Instances(inst.dataset());

		if (mWindowSearch == null) {
			if (this.nearestNeighbourSearchOption.getChosenIndex()== 0)
				mWindowSearch = new LinearNNSearch(mWindow);
			else
				mWindowSearch = new KDTree(mWindow);
		}
		
		if (mLUSearch == null) {
			if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
				mLUSearch = new LinearNNSearch(mWindow);  
			} else {
				mLUSearch = new KDTree(mWindow);
			}
		}

		if (mHashInstancesL == null)
			mHashInstancesL = new Hashtable<>();
			
		if (mShiftRegisters == null)
			mShiftRegisters = new Hashtable<>();

		try {
			k = (int)Math.ceil(mBeta.getValue() * mWindow.numInstances());
			mWindowSearch.kNearestNeighbours(inst, k);
			distNN = mWindowSearch.getDistances();
			
			// Get the instances in L u U into the radius
			if (distNN.length > 0)
				nn = getNNWithinRadius(inst, distNN[distNN.length - 1]);
	
			// Add inst to L
			mWindow.add(inst);
			mInstancesLU.add(inst);
			mHashInstancesL.put(inst.toString(), true);
			mWindowSearch.update(inst);
			mLUSearch.update(inst);
			
			if (distNN.length > 0) {
				// for each neighbor 
				for (int i = 0; i < nn.size(); i++) {
					// Compute SRei
					shiftReg = mShiftRegisters.get(nn.get(i).toString());
					
					if (shiftReg == null)
						shiftReg = new ArrayList<Boolean>(mRegisterLenght.getValue());
					
					// Store SRei
					shiftReg.add(agree(inst, nn.get(i)));
					
					if (shiftReg.size() > mRegisterLenght.getValue())
						shiftReg.remove(0);
					
					mShiftRegisters.put(nn.get(i).toString(), shiftReg);
					
					x = successes(shiftReg);
					
					if (calcBound (true, x, shiftReg.size()) > 
					     mAgreementAcceptance.getValue() &&
					      !mHashInstancesL.get(nn.get(i).toString())) {
						mHashInstancesL.put(nn.get(i).toString(), true);
						mWindow.add(nn.get(i));
					}
					
					if (calcBound (false, x, shiftReg.size()) < 
					     mInactivationThreshold.getValue() &&
					      mHashInstancesL.get(nn.get(i).toString())) {
						mHashInstancesL.put(nn.get(i).toString(), false);
						removeFromWindow(nn.get(i));
					}
				}
			}

		} catch (Exception e) {
			System.err.println("Error: kNN search failed.");
			e.printStackTrace();
		}
	}

	/**
	 * Computes and returns the instances which the distance between them and
	 * inst is lower than radius. 
	 * 
	 * @param inst center of the hypersphere.
	 * @param radius radius of the hypersphere.
	 * @return Those instances into the hypersphere.
	 */
	public ArrayList<Instance> getNNWithinRadius (Instance inst, double radius) {
		ArrayList<Instance> nn = new ArrayList<>();
		
		for (int i = 0; i < mInstancesLU.numInstances(); i++)
			if (mLUSearch.getDistanceFunction().
			       distance(mInstancesLU.instance(i), inst) <= radius)
				nn.add(mInstancesLU.instance(i));
		
		return nn;
	}

	/**
	 * Remove the instance from the window.
	 * 
	 * @param inst instance to remove from the window.
	 * @throws Exception if an exception in search occurs.
	 */
	private void removeFromWindow(Instance inst) throws Exception {
		for (int i = 0; i < mWindow.numInstances(); i++) {
			if (LWF.compare(inst, mWindow.instance(i)) == 0) {
				mWindow.delete(i);
				mWindowSearch.setInstances(mWindow);
				return;
			}
		}
	}
	
	/**
	 * Checks the consistency between two instances. 
	 * 
	 * @param newInst New instance to analyze.
	 * @param oldInst Old instance already stored.
	 * @return True if both instances has the same class value.
	 */
	private boolean agree (Instance newInst, Instance oldInst) {
		if (newInst.classValue() == oldInst.classValue())
			return true;
		
		return false;
	}
	
	/**
	 * Computes the number of trues that are present in the register.
	 * 
	 * @param shiftReg Register.
	 * @return Number of trues in the register.
	 */
	private double successes (List<Boolean> shiftReg) {
		int num = 0;
		
		for (boolean b : shiftReg)
			if (b)
				num++;
		
		return num;
	}

	/**
	 * Returns the upper/lower boundary.
	 * 
	 * @param lower True for lower boundary, false for upper.
	 * @param x 
	 * @param n
	 * @return Lower/Upper boundary.
	 */
	private double calcBound (boolean lower, double x, double n) {
		double p = x / n;
		double f = getConfValue();
		
		if (lower)
			return p - f * Math.sqrt((p * (1 - p)) / n);
		
		return p + f * Math.sqrt((p * (1 - p)) / n);
	}

	/**
	 * Returns the factor according to the confidence value.
	 * 
	 * @return Confidence value.
	 */
	private double getConfValue () {
		
		switch (mConfidenceValue.getChosenIndex()) {
			case 0:
				return PERC_99;
			case 1:
				return PERC_95;
			case 2:
				return PERC_90;
		}
		
		return PERC_80;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		Instances neighbours;
		double v[] = new double[mNumClasses];
		
		try {
			if (mWindow.numInstances() > 0) {
				neighbours = mWindowSearch.kNearestNeighbours(inst,Math.min(kOption.getValue(),
						                                        mWindow.numInstances()));
				
				for(int i = 0; i < neighbours.numInstances(); i++)
					v[(int)neighbours.instance(i).classValue()]++;
			}
		} catch(Exception e) {
			System.err.println("Error: kNN search failed.");
			e.printStackTrace();
			return new double[inst.numClasses()];
		}
		return v;
	}
	
	@Override
	public void resetLearningImpl() {
		super.resetLearningImpl();
		mWindowSearch = null;
		mInstancesLU = null;
		mLUSearch = null;
		mHashInstancesL = null;
		mShiftRegisters = null;
	}

	public String getPurposeString() {
		
		return "Prediction Error Context Switching (PECS).";
	}
	
}
