/*
 * ANNCADTree.java
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

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.streams.generators.HyperplaneGenerator;

/**
 * Implementation of the members for Adaptive NN Classification 
 * Algorithm for Data-streams (ANNCAD). Presented in:
 * Law, Y. N., & Zaniolo, C. (2005). An adaptive nearest neighbor 
 * classification algorithm for data streams. In Knowledge Discovery 
 * in Databases: PKDD 2005 (pp. 108-120). Springer Berlin Heidelberg.
 * <p>
 * Notes:
 * <p>
 * It is not prepared for nominal attributes.<br>
 * The memory run out's procedure is not implemented.<br> 
 * <p>
 * Valid options are:
 * <p>
 * -d depth of the tree <br>
 * -l lower boundary for the grid <br>
 * -u upper boundary for the grid <br>
 * -t threshold for deciding if go down in the tree or not <br>
 * -f forgetting factor <br>
 * -s maximum shift of the grid <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160606
 */
public class ANNCADTree extends AbstractClassifier {

	private static final long serialVersionUID = -7323723076168696015L;

	public IntOption mMaxDepthTree = new IntOption("maxDepthTree", 'd',
	         "Max. depth of the tree", 4, 1, Integer.MAX_VALUE);

	public FloatOption mLowerBoundary = new FloatOption("lowerBoundary", 'l',
	         "Lower boundary of the grid", 0, Integer.MIN_VALUE, Integer.MAX_VALUE);

	public FloatOption mUpperBoundary = new FloatOption("upperBoundary", 'u',
	         "Upper boundary of the grid", 1, Integer.MIN_VALUE, Integer.MAX_VALUE);

	public FloatOption mThreshold = new FloatOption("threshold", 't', "Threshold", 
	         1, 0, 1);

	public FloatOption mForgetting = new FloatOption("forgetting", 'f', 
	         "Forgetting factor", 0.98, 0, 1);
	
	public FloatOption mShift = new FloatOption("shift", 's',
	         "Maximum random shift of the grid", 0.5, 0, 1);

	private Block mBlock;
	
	private int mClasses;

	@Override
	public double[] getVotesForInstance(Instance inst) {
		// From finest to coarsest.
		if (mBlock != null)
			return mBlock.getVotesForInstance (inst);
		
		return new double[inst.numClasses()];
	}
	
	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			mClasses = context.numClasses();
		} catch (Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	public void resetLearningImpl() {
		mBlock = null;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (mClasses == 0)
			mClasses = inst.numClasses();
		
		if (mBlock == null) {
			Random rand = new Random (randomSeed);
			double[] lowBoundaries = new double[inst.numAttributes() - 1];
			double[] uppBoundaries = new double[inst.numAttributes() - 1];
			
			for (int i = 0, j = 0; i < inst.numAttributes(); i++) {
				if (i != inst.classIndex()) {
					lowBoundaries[j] = mLowerBoundary.getValue() - 
					  (mShift.getValue() * rand.nextDouble());
					uppBoundaries[j] = mUpperBoundary.getValue() + 
					  (mShift.getValue() * rand.nextDouble());
					
					j++;
				}
			}
			
			mBlock = new Block (mClasses, inst.numAttributes() - 1, 0, 
			           mMaxDepthTree.getValue(), lowBoundaries, uppBoundaries);
		}

		// Add the new instance to the block's structure.
		mBlock.add(inst);
		
		// Exponential forgetting.
		if (mForgetting.getValue() != 1)
			mBlock.exponentialForgetting(mForgetting.getValue());
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

		return "ANNCAD: Adaptive NN Classification Algorithm for Data-streams.";
	}

	@Override
	public boolean isRandomizable() {
		
		return true;
	}
	
	/**
	 * Block that contains the information that ANNCAD needs.
	 */
	private class Block implements Serializable {
		
		private static final long serialVersionUID = -8597363231184332155L;

		/**
		 * Number of instances of each class in the block.
		 */
		private double mVotesPerClass[];
		
		/**
		 * Depth's level of the block.
		 */
		private int mCurrentDepth;
		
		/**
		 * Maximum depth of blocks.
		 */
		private int mMaxDepth;
		
		/**
		 * Center of the block. Used for the NN calculations.
		 */
		private double mCentre[];
		
		/**
		 * Space's lower boundaries.  
		 */
		private double mLowerBound[];
		
		/**
		 * Space's upper boundaries.  
		 */
		private double mUpperBound[];

		/**
		 * Children of the block.
		 */
		private HashMap<String, Block> mChilds;
		
		/**
		 * Constructor of the tree.
		 * 
		 * @param numClasses Number of classes.
		 * @param numAtt Number of attributes.
		 * @param currDepth Depth's level of the tree.
		 * @param maxDepth Maximum level that the tree can grow.
		 * @param lower Lower boundary in the space.
		 * @param upper Upper boundary in the space.
		 */
		public Block (int numClasses, int numAtt, int currDepth, int maxDepth,
		                double lower, double upper) {
			mVotesPerClass = new double[numClasses];
			mChilds = new HashMap<>(4);
			mCurrentDepth = currDepth;
			mMaxDepth = maxDepth;
			mLowerBound = new double[numAtt];
			mUpperBound = new double[numAtt];
			mCentre = new double[numAtt];

			for (int i = 0; i < numAtt; i++) {
				mLowerBound[i] = lower;
				mUpperBound[i] = upper;
				mCentre[i] = lower + (upper - lower) / 2;
			}
		}
		
		/**
		 * Constructor of the tree.
		 * 
		 * @param numClasses Number of classes.
		 * @param numAtt Number of attributes.
		 * @param currDepth Depth's level of the tree.
		 * @param maxDepth Maximum level that the tree can grow.
		 * @param lower Lower boundaries in the space, one for each dimension.
		 * @param upper Upper boundaries in the space, one for each dimension
		 */
		public Block(int numClasses, int numAtt, int currDepth, 
		              int maxDepth,double lower[], double upper[]) {
			this(numClasses, numAtt, currDepth, maxDepth, 0, 0);
			
			for (int i = 0; i < numAtt; i++) {
				mLowerBound[i] = lower[i];
				mUpperBound[i] = upper[i];
				mCentre[i] = lower[i] + ((upper[i] - lower[i]) / 2);
			}
		}
		
		/**
		 * Adds an instance to the tree.
		 * 
		 * @param inst Instance to add.
		 */
		public void add (Instance inst) {
			mVotesPerClass[(int)inst.classValue()]++;
			
			if (mCurrentDepth != mMaxDepth) {
				double[] lowBoundaries = new double[inst.numAttributes() - 1];
				double[] uppBoundaries = new double[inst.numAttributes() - 1];
				String key = calcKey(inst, mCurrentDepth + 1, 
				                       lowBoundaries, uppBoundaries);
				
				Block child = mChilds.get(key);
				
				if (child == null) {
					child = new Block (mClasses, inst.numAttributes() - 1, 
					                    mCurrentDepth + 1, mMaxDepth,
					                    lowBoundaries, uppBoundaries);
					mChilds.put(key, child);
				}
				
				child.add(inst);
			}			
		}
		
		/**
		 * Computes the class predicted by the block.
		 */
		private int getPredClass () {
			double contMaj1 = 0, contMaj2 = 0;
			int maj1 = 0;
			
			// Look for the 1st & 2nd majority classes. 
			contMaj1 = mVotesPerClass[0];
			
			for (int i = 1; i < mVotesPerClass.length; i++) {
				if (mVotesPerClass[i] > contMaj1) {
					contMaj2 = contMaj1;
					maj1 = i;
					contMaj1 = mVotesPerClass[i];
				} else if (mVotesPerClass[i] > contMaj2) {
					contMaj2 = mVotesPerClass[i];
				}
			}
			
			if (( 1 - (contMaj2 / contMaj1)) >= mThreshold.getValue())
				return maj1;
			else
				return -1;
		}
		
		/**
		 * Computes the tree's key for an instance.
		 * 
		 * @param inst Instance.
		 * @param depth Depth of the tree.
		 * @param lowerBoundaries Lower boundaries of the cell.
		 * @param upperBoundaries Upper boundaries of the cell.
		 * @return Tree's key.
		 */
		private String calcKey (Instance inst, int depth, double lowerBoundaries[],
		                          double upperBoundaries[]) {
			String combKey = new String();
			
			// Compute the z-value
			for (int i = 0, j = 0; i < inst.numAttributes(); i++) {
				if (i != inst.classIndex()) {
					// Compute the cell.
					if (inst.value(i) < mCentre[j]) {
						lowerBoundaries[j] = mLowerBound[j];
						upperBoundaries[j] = mCentre[j];
						combKey += "0";
					}
					else {
						lowerBoundaries[j] = mCentre[j];
						upperBoundaries[j] = mUpperBound[j];
						combKey += "1";
					}
					
					j++;
				}
			}
			
			return combKey;
		}
		
		/**
		 * Computes the votes for an instance.
		 * 
		 * @param inst Instance.
		 * @return Array with the votes.
		 */
		public double[] getVotesForInstance (Instance inst) {
			double[] lowBoundaries = new double[inst.numAttributes() - 1];
			double[] uppBoundaries = new double[inst.numAttributes() - 1];
			double[] v = new double[mClasses];
			int predClass;
			
			predClass = getPredClass();
			
			if (predClass != -1) {
				v[predClass]++;
				
				return v;
			}
			
			// If it's a block of the last level.
			if (mMaxDepth == mCurrentDepth) {
				for (int i = 0; i < mClasses; i++)
					v[i] = mVotesPerClass[i];
				
				return v;
			}
			
			// Obtain the key of inst.
			String key = calcKey(inst, mCurrentDepth + 1, lowBoundaries, uppBoundaries);
			Block child = mChilds.get(key);
			
			// Calc the NN cell.
			if (child == null) {
				double tmpDist, minDist = Double.MAX_VALUE;
				Block nearestCell = null;
				Iterator<Entry<String, Block>> it = mChilds.entrySet().iterator();
			    while (it.hasNext()) {
			    	Entry<String, Block> pair = (Entry<String, Block>)it.next();
			    	tmpDist = pair.getValue().distance (inst);
			    	
					if (tmpDist < minDist) {
						minDist = tmpDist;
						nearestCell = pair.getValue();
					}
			    }
			    
				return nearestCell.getVotesForInstance(inst);
			}
			
			return child.getVotesForInstance(inst);
		}
		
		/**
		 * Computes the distance between the cell and the instance.
		 * 
		 * @param inst Instance.
		 * @return Euclidean distance between inst and the cell.
		 */
		public double distance (Instance inst) {
			double diff = 0;
			
			for (int j = 0, i = 0; i < inst.numAttributes(); i++) {
				if (i != inst.classIndex()) {
					diff += Math.pow(mCentre[j] - inst.value(i), 2); 
					j++;
				}
			}
			
			return Math.sqrt(diff);
		}
		
		/**
		 * Exponential forgetting: the prediction is updated by multiplying by
		 * lambda. Really time-consuming task.
		 * 
		 * @param lambda Forgetting factor.
		 */
		public void exponentialForgetting(double lambda) {
			// Propagate the forgetting to its child.
			if (mMaxDepth != mCurrentDepth) {
				Iterator<Entry<String, Block>> it = mChilds.entrySet().iterator();
				while (it.hasNext()) {
					((Entry<String, Block>) it.next()).getValue()
					   .exponentialForgetting(lambda);
				}
			}
			
			for (int i = 0; i < mVotesPerClass.length; i++)
				mVotesPerClass[i] *= lambda;
		}
	}
}
