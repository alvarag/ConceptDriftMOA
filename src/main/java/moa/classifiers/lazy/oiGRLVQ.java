/*
 * OIGRLVQ.java
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

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.EuclideanDistance;
import moa.classifiers.lazy.neighboursearch.LinearNNGranuleSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;

/**
 * Implementation of oi-GRLVQ. Presented in: 
 * Cruz-Vega, I., & Escalante, H. J. (2016). 
 * An online and incremental GRLVQ algorithm for 
 * prototype generation based on granular computing. 
 * Soft Computing, 1-14.
 * <p>
 * Notes:
 * <p>
 * Only works with numeric attributes. <br>
 * <p>
 * Valid options are:
 * 
 * @author Álvar Arnaiz-González
 * @version 20160525
 */
public class oiGRLVQ extends AbstractClassifier {

	private static final long serialVersionUID = 7395371590666920530L;

	public FloatOption mLearningRate = new FloatOption("learningRate", 'l', 
            "Learning rate", 0.1, 0, 1);

	public FloatOption mLearningDec = new FloatOption("learningDecreasing", 'd', 
	        "Learning decreasing rate", 0, 0, 1);

	public FloatOption mRelevInit = new FloatOption("relevancesInit", 'i', 
	        "Relevances initialization", 0.1, 0, 1);

	public IntOption mInstsPerBatch = new IntOption("numInstPerBatch", 'b', 
	        "Number of prototypes of each mini-batch", 500, 1, Integer.MAX_VALUE);

	public IntOption mDesiredReduction = new IntOption("desiredReduction", 'r', 
	        "Desired reduction per class (for pruning)", 25, 0, 100);

	/**
	 * Euclidean distance for computations.
	 */
	private EuclideanDistance mEucDist;
	
	/**
	 * List of granules.
	 */
	private ArrayList<Granule> mGranules;

	/**
	 * Mini-Batch used by the algorithm.
	 */
	private Instances mMiniBatch;
	
	/**
	 * ρ = 1 − 0.01 x sqrt(att_num)
	 */
	private double mP;
	
	/**
	 * If the initialization of granules has been made. 
	 */
	private boolean mInitGranules;
	
    /**
     * Current learning rate. It decreases.
     */
    private double mCurrLearnRate;
    
	/**
	 * Instances read.
	 */
	private long mInstancesRead;
	
	/**
	 * Relevance terms (&lambda;)
	 */
	private double[] mRelTerms;
	
	@Override
	public boolean isRandomizable() {
		
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		NearestNeighbourSearch search;
		double v[] = new double[inst.numClasses()];
		
		try {
			if (mGranules != null) {	
				search = new LinearNNGranuleSearch(mGranules, mMiniBatch);  
				Instance neighbours = search.nearestNeighbour(inst);
				v[(int)neighbours.classValue()]++;
			}
		} catch(Exception e) {
			return new double[inst.numClasses()];
		}
		
		return v;
	}

	@Override
	public void resetLearningImpl() {
		mGranules = null;
		mMiniBatch = null;
		mRelTerms = null;
		mInitGranules = false;
		mInstancesRead = 0;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (mEucDist == null) {
			mEucDist = new EuclideanDistance(mMiniBatch);
			mEucDist.setDontNormalize(true);
		}
		
		if (mMiniBatch == null)
			mMiniBatch = new Instances(inst.dataset(), 
			                  mInstsPerBatch.getValue());
		
		if (mGranules == null)
			mGranules = new ArrayList<>();
			
		if (mRelTerms == null) {
			mRelTerms = new double[inst.numAttributes() - 1];
			for (int i = 0; i < mRelTerms.length; i++)
				mRelTerms[i] = mRelevInit.getValue();
		}
			
		if (mP == 0)
			mP = 1 - (0.01 * Math.sqrt(inst.numAttributes() - 1));
		
		// Add the new instance.
		mInstancesRead++;
		mMiniBatch.add(inst);
		updateDifferences(inst);
		
		// Initialization step: one granule per class.
		if (!mInitGranules) {
			checkInitGranules(inst);

			// Initialize learning rate.
			mCurrLearnRate = mLearningRate.getValue();
			
			return;
		}
		
		// Online learning.
		if (mMiniBatch.numInstances() < mInstsPerBatch.getValue()) {
			onlineLearning (inst);
		}
		// Updates relevances with mini-batch.
		else {
			updateRelevances();
			mMiniBatch = new Instances(inst.dataset(), 
			                   mInstsPerBatch.getValue());
			
			// TODO Pruning here?
			if (mDesiredReduction.getValue() != 0)
				pruning ();
		}

		// TODO Pruning here?
//		if (mDesiredReduction.getValue() != 0)
//			pruning ();
	}
	
	/**
	 * Check if the instance should be added as a new granule.
	 * 
	 * @param inst New instance.
	 */
	private void checkInitGranules (Instance inst) {
		int[] classes = countInstPerClass(inst.numClasses());
		
		mInitGranules = true;
		
		// Add a new granule if doesn't exist a granule of the inst's class.
		if (classes[(int)inst.classValue()] == 0) {
			mGranules.add(new Granule(inst));
			classes[(int)inst.classValue()]++;
		}
		
		// Have we got a granule per class?
		for (int i = 0; i < classes.length; i++)
			if (classes[i] == 0)
				mInitGranules = false;
	}

	/**
	 * Computes the number of granules per class.
	 * 
	 * @param classes Number of classes.
	 * @return Number of granules per class.
	 */
	private int[] countInstPerClass(int classes) {
		int[] granPerClass = new int[classes];
		
		// Count granules/classes
		for (Granule granule : mGranules)
			granPerClass[granule.classValue()]++;
		
		return granPerClass;
	}
	
	/**
	 * Online learning, for each instance the granules are reweighted or a
	 * new one is created.
	 * 
	 * @param inst New instance.
	 */
	private void onlineLearning(Instance inst) {
		double[] distPosNeg = new double[2];
		int[] indexPosNeg = new int[2];
		
		// Find the nearest granules.
		findNearestGranules(inst, indexPosNeg, distPosNeg);
		
		// Update existing granules.
		if (mGranules.get(indexPosNeg[0]).similarity(inst, mInstancesRead) > mP) {
			mGranules.get(indexPosNeg[0]).updateWeights(inst, mCurrLearnRate, true);
			mGranules.get(indexPosNeg[1]).updateWeights(inst, mCurrLearnRate, false);
			
			mGranules.get(indexPosNeg[0]).addFreqUsage ();
			mGranules.get(indexPosNeg[1]).addFreqUsage ();
			
			mCurrLearnRate -= mLearningDec.getValue();
		}
		// Add new granule.
		else {
			mGranules.add(new Granule(inst));
		}
	}
	
	/**
	 * Computes the instances' nearest granules of the same and distinct 
	 * class and their distances.
	 * 
	 * @param inst Instance.
	 * @param indexPosNeg Array with the indexes of the nearest granule 
	 *        of the same[0]/other[1] class.
	 * @param distPosNeg Array with distances to the nearest granule
	 *        of the same[0]/other[1] class.
	 */
	private void findNearestGranules (Instance inst, int[] indexPosNeg, 
	                                   double[] distPosNeg) {
		double dist;
		
		distPosNeg[0] = Double.MAX_VALUE;
		distPosNeg[1] = Double.MAX_VALUE;
		
		// Find the nearest positive/negative granules.
		for (int i = 0; i < mGranules.size(); i++) {
			dist = mEucDist.distance(mGranules.get(i).instance(), inst);
			if (inst.classValue() == mGranules.get(i).classValue()) {
				if (dist < distPosNeg[0]) {
					distPosNeg[0] = dist;
					indexPosNeg[0] = i;
				}
			}
			else {
				if (dist < distPosNeg[1]) {
					distPosNeg[1] = dist;
					indexPosNeg[1] = i;
				}
			}
		}
	}
	
	/**
	 * Updates the relevances of the granules with the instances of the
	 * mini-batch.
	 */
	private void updateRelevances() {
		double[] distPosNeg = new double[2];
		int[] indexPosNeg = new int[2];
		
		for (int i = 0; i < mMiniBatch.numInstances(); i++) {
			// Find the nearest granules.
			findNearestGranules(mMiniBatch.instance(i), indexPosNeg, distPosNeg);

			// Update their relevances.
			updateRel(mMiniBatch.instance(i), mGranules.get(indexPosNeg[0]), 
			            mGranules.get(indexPosNeg[1]), distPosNeg[0], distPosNeg[1]);
		}
		
		// Normalize to ensure ||lambda|| = 1
		normRelTer();
	}
	
	/**
	 * Updates the relevance terms.
	 * 
	 * @param inst Instance that triggers the update.
	 * @param pos Nearest granule of inst of the same class.
	 * @param neg Nearest granule of inst of the other class.
	 * @param disPos Distance from inst to pos.
	 * @param disNeg Distance from inst to neg.
	 */
	public void updateRel(Instance inst, Granule pos, Granule neg,
	                      double disPos, double disNeg) {
		double p, n, sqDis;

		sqDis = Math.pow(disPos + disNeg, 2);

		for (int j = 0, i = 0; i < inst.numAttributes(); i++) {
			if (i != inst.classIndex()) {
				p = (disNeg * Math.pow(mEucDist.sqDifference(i, 
				               inst.value(i), pos.value(i)), 2))
				               / sqDis;

				n = (disPos * Math.pow(mEucDist.sqDifference(i, 
				               inst.value(i), neg.value(i)), 2))
				               / sqDis;
				
				// Don't accumulate negative values.
				mRelTerms[j] = Math.max(0, mRelTerms[j] - (1 * (p - n)) 
				                           * mLearningRate.getValue());
				j++;
			}
		}
	}
	
	/**
	 * Normalize the relevance term to ensure: ||&lambda;|| = 1.
	 */
	public void normRelTer () {
		double norm = 0;
		
		for (int i = 0; i < mRelTerms.length; i++)
			norm += Math.pow(mRelTerms[i], 2);
		
		norm = Math.sqrt(norm);
		
		for (int i = 0; i < mRelTerms.length; i++)
			mRelTerms[i] /= norm;
	}
	
	/**
	 * Pruning the granules.
	 */
	private void pruning () {
		int[] instPerClass = countInstPerClass(mMiniBatch.numClasses());
		double[] freqUsage, indexes;
		int rem;
		
		for (int i = 0; i < instPerClass.length; i++) {
			freqUsage = new double[instPerClass[i]];
			indexes = new double[instPerClass[i]];
			
			for (int j = 0, k = 0; j < mGranules.size(); j++) {
				if ((int)mGranules.get(j).classValue() == i) {
					indexes[k] = j;
					freqUsage[k] = mGranules.get(j).getFreqUsage();
					k++;
				}
			}
			
			NearestNeighbourSearch.quickSort(indexes, freqUsage, 0, 
			                                  freqUsage.length - 1);
			
			// Desired reduction
			rem = instPerClass[i] - (instPerClass[i] * 
			        mDesiredReduction.getValue()) / 100;
			
			if (rem > 0) {
				for (int j = instPerClass[i] - 1; j >= rem; j--) {
					mGranules.remove((int)indexes[j]);
				}
			}
		}
	}

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			mMiniBatch = new Instances(context, 0);
			mMiniBatch.setClassIndex(context.classIndex());
		} catch (Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	public String getPurposeString() {
		
		return "oi-GRLVQ.";
	}

	/**
	 * Adds the new instance to the granules' differences used
	 * for the radius.
	 * 
	 * @param inst Instance for radius computation.
	 * @param instNum Number of instances read until now
	 * @return Radius after new instance arrives.
	 */
	public void updateDifferences (Instance inst) {
		for (Granule gran : mGranules)
			gran.addDiffRadius(mEucDist.distance(inst, 
			                     gran.instance()));
	}
	
	/**
	 * Granule for the granular computing.<br>
	 * The center of the granule is represented as an instance.
	 */
	public class Granule {
		
		/**
		 * Center of the granule.
		 */
		private Instance mWeights;
		
		/**
		 * Radius of the granule (&sigma;).
		 */
		private double mRadius;
		
		/**
		 * Frequency usage.
		 */
		private long mFreqUsage;
		
		public Granule (Instance inst) {
			mWeights = new DenseInstance(inst);
			
			mFreqUsage = 1;
			mRadius = 0;
		}
		
		/**
		 * Similarity between the granule and the instance.
		 * 
		 * @param inst Instance.
		 * @param instNumber Number of instances read until now
		 * @return Similarity.
		 */
		public double similarity (Instance inst, long instNumber) {
			double mu, up = 0;
			
			for (int i = 0, j = 0; i < inst.numAttributes(); i++) {
				if (i != inst.classIndex()) {
					up += Math.pow(inst.value(i) - mWeights.value(j), 2)
					         * mRelTerms[j];
					j++;
				}
			}
			
			mu = Math.exp(-up / Math.pow((mRadius/instNumber), 2));

			return mu;
		}

		/**
		 * Add a new difference for the radius calculation.
		 * 
		 * @param diff Difference.
		 */
		public void addDiffRadius (double diff) {
			mRadius += diff;
		}

		/**
		 * Returns the value of the weight.
		 * 
		 * @param pos Index of the weight.
		 * @return Weight at the position pos.
		 */
		public double value (int pos) {
			
			return mWeights.value(pos);
		}
		
		/**
		 * Updates the weights according to the instance.
		 * 
		 * @param inst Instance.
		 * @param learnRate Learning rate.
		 * @param inc Increases the weight if is True, decreases otherwise.
		 */
		public void updateWeights (Instance inst, double learnRate, boolean inc) {
			double diff;
			
			for (int i = 0, j = 0; i < inst.numAttributes(); i++) {
				if (i != inst.classIndex()) {
					diff = inst.value(i) - value(j);
					if (inc)
						mWeights.setValue(j, value(j) + (learnRate * diff));
					else
						mWeights.setValue(j, value(j) - (learnRate * diff));
					
					j++;
				}
			}
		}
		
		/**
		 * Increases the frequency usage by adding one.
		 */
		public void addFreqUsage () {
			mFreqUsage++;
		}

		/**
		 * Returns the frequency usage of the granule.
		 * 
		 * @return Frequency usage.
		 */
		public long getFreqUsage () {
			
			return mFreqUsage;
		}
		
		/**
		 * Returns the class value of the granule.
		 * 
		 * @return Class value of the granule.
		 */
		public int classValue () {
			
			return (int)value(mWeights.classIndex());
		}

		/**
		 * Returns the center of the granule.
		 * 
		 * @return The instance.
		 */
		public Instance instance () {
			
			return mWeights;
		}
	}	
}
