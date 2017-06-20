/*
 * OLVQ.java
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

import java.util.Random;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.EuclideanDistance;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;

/**
 * Implementation of online-LVQ. Presented in: 
 * Bharitkar, S., & Filev, D. (2001, August). 
 * An online learning vector quantization 
 * algorithm. In ISSPA (pp. 394-397).
 * <p>
 * Some of the code has been extracted from Weka
 * clusterer LVQ.
 * <p>
 * Notes:
 * <p>
 * Only works with numeric attributes. <br>
 * Until the first batch offline learning, the predicted
 * value is based on the sliding window.<br>
 * Initially the code vectors are centered at 0.5 and there
 * are the same number for each class.<br> 
 * <p>
 * Valid options are:
 * n nearest neighbour search to use <p>
 * k number of code vectors <p>
 * i instances for the first batch online learning <p>
 * e number of epochs <p>
 * l learning rate <p>
 * w window limit of the training window <p>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160524
 */
public class OLVQ extends AbstractClassifier {

	private static final long serialVersionUID = 2963502550035525519L;

	public IntOption mNumCodeVect = new IntOption("codeVectorsNumber", 'k', 
	                  "Code vectors number", 100, 1, Integer.MAX_VALUE);

	public IntOption mInitialLearning = new IntOption("initialLearning", 'i', 
	                  "Initial learning", 200, 1, Integer.MAX_VALUE);

	public IntOption mNumEpochs = new IntOption("epochsNumber", 'e', 
	                  "Epochs number", 100, 1, Integer.MAX_VALUE);

	public FloatOption mLearningRate = new FloatOption("learningRate", 'l', 
	                    "Learning rate", 0.1, 0, 1);

	public FloatOption mLearningDec = new FloatOption("learningDecreasing", 'd', 
	                    "Learning decreasing rate", 0, 0, 1);

	public IntOption mWindowLimit = new IntOption("windowLimit", 'w', 
	         "Limit of the training window", 300, 1, Integer.MAX_VALUE);

	/**
	 * Sliding window for training.
	 */
	private Instances mTrainWindow;
	
	/**
	 * Code vectors of LVQ.
	 */
	private Instances mCodeVectors;
	
	/**
	 * Wether is in offline or online step.
	 */
	private boolean mOffline;

	/**
	 * Index of the code vectors.
	 */
    private int[] mClusterList;
    
    /**
     * Epochs of the online step.
     */
    private int mOnlineEpochs;
    
    /**
     * Current learning rate. It decreases.
     */
    private double mCurrLearnRate;
    
    private long mInstancesRead;
    
	@Override
	public boolean isRandomizable() {
		
		return true;
	}

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			mTrainWindow = new Instances(context, 0);
			mTrainWindow.setClassIndex(context.classIndex());
			mCodeVectors = new Instances(context, 0);
			mCodeVectors.setClassIndex(context.classIndex());
		} catch (Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
		}
	}

	/**
	 * Predict the class of the new instance.
	 * Until the first batch offline learning, the prediction is made
	 * by the nearest neighbour classifier over the sliding window.
	 * 
	 * @param inst Instance to predict.
	 * @return Predicted value.
	 */
    public double[] getVotesForInstance(Instance inst) {
		double v[] = new double[inst.numClasses()];
		
		try {
			NearestNeighbourSearch search = new LinearNNSearch(mCodeVectors);
			
			if (mCodeVectors != null) {	
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
		mTrainWindow = null;
		mCodeVectors = null;
		mClusterList = null;
		mOffline = true;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (mTrainWindow == null) {
			mTrainWindow = new Instances(inst.dataset());
		}
		
		if (mCodeVectors == null) {
			mCodeVectors = new Instances(inst.dataset());
		}
		
		mInstancesRead++;
		
		// a) Offline learning
		if (mOffline) {
			if (mTrainWindow.numInstances() < mInitialLearning.getValue()) {
				mTrainWindow.add(inst);
				mCodeVectors.add(inst);
			} else {
				mOffline = false;
				mOnlineEpochs = mNumEpochs.getValue();
				
				// Initialize learning rate.
				mCurrLearnRate = mLearningRate.getValue();

				buildOffline();
			}
		}
		// b) Online learning
		else {
			// Update the training window.
			mTrainWindow.add(inst);
			if (mWindowLimit.getValue() <= mTrainWindow.numInstances())
				mTrainWindow.delete(0);
			
			onlineLearning (inst);
			
			if (mOnlineEpochs > 1)
				mOnlineEpochs -= 1;
			
			if (mInstancesRead > 5000) {
				System.out.println ("Results:");
				for (int i = 0; i < mCodeVectors.numInstances(); i++)
					System.out.println (mCodeVectors.instance(i));
			}

		}
	}

	/**
	 * First training: offline.
	 */
	private void buildOffline() {
		int winningNeuron;

		mCodeVectors = initClusters();

		// init the pointList (used in EuclideanDistance.closestPoint)
		mClusterList = new int[mNumCodeVect.getValue()];
		
		for (int i = 0; i < mClusterList.length; i++)
			mClusterList[i] = i;
		
		// init euclidean distance
		EuclideanDistance distance = new EuclideanDistance(mCodeVectors);
		distance.setDontNormalize(true);

		try {
			for (int epoch = 0; epoch < mNumEpochs.getValue(); epoch++) {
				for (int instance = 0; instance < mTrainWindow.numInstances(); instance++) {
					winningNeuron = distance.closestPoint(mTrainWindow.get(instance), 
					                                      mCodeVectors, mClusterList);
					
					// update the weights
					for (int j = 0; j < mCodeVectors.numAttributes(); j++) {
						if (j != mCodeVectors.classIndex()) {
							double diff = mCurrLearnRate
							               * (mTrainWindow.get(instance).value(j) - 
							                   mCodeVectors.get(winningNeuron).value(j));
							
							if (!Double.isNaN(diff)) {
								updateCodeVector(winningNeuron, instance, j, diff);
							}
						}
					}
				}
				mCurrLearnRate -= mLearningDec.getValue();
			} 
		} catch (Exception e) {
			System.err.println("Error: buildOffline.");
			e.printStackTrace();
		}
	}
	
	/**
	 * Online training.
	 * 
	 * @param inst New instance.
	 */
	private void onlineLearning (Instance inst) {
		double dist, sumDist;
		int winningNeuron;

		// init euclidean distance
		EuclideanDistance distance = new EuclideanDistance(mCodeVectors);
		distance.setDontNormalize(true);

		try {
			for (int epoch = 0; epoch < mOnlineEpochs; epoch++) {
				for (int instance = 0; instance < mTrainWindow.numInstances(); instance++) {
					winningNeuron = distance.closestPoint(mTrainWindow.get(instance), 
					                                       mCodeVectors, mClusterList);

					dist = sumDist = 0;
					for (int i = 0; i < mCodeVectors.numInstances(); i++) {
						if (i == winningNeuron)
							dist = distance.distance(inst, mCodeVectors.instance(i));
						else
							sumDist += distance.distance(inst, mCodeVectors.instance(i));
					}
					
					// update the weights
					for (int j = 0; j < mCodeVectors.numAttributes(); j++) {
						if (j != mCodeVectors.classIndex()) {
							double diff = mCurrLearnRate
							              * (mTrainWindow.get(instance).value(j) - 
							                  mCodeVectors.get(winningNeuron).value(j))
							                  * (dist / sumDist);
							
							if (!Double.isNaN(diff)) {
								updateCodeVector(winningNeuron, instance, j, diff);
							}
						}
					}
				}
				mCurrLearnRate -= mLearningDec.getValue();
			}
		} catch (Exception e) {
			System.err.println("Error: onlineLearning.");
			e.printStackTrace();
		}
	}

	/**
	 * Updates the position of a code vector.
	 * 
	 * @param winningNeuron Index of the code vector.
	 * @param instance Instance that has triggered the update.
	 * @param att Attribute to update.
	 * @param diff Difference to add/subtract to the code vector.
	 */
	private void updateCodeVector(int winningNeuron, int instance, int att,
	                               double diff) {
		// add
		if (mCodeVectors.get(winningNeuron).classValue() == 
		      mTrainWindow.get(instance).classValue())
			mCodeVectors.get(winningNeuron).setValue(att,
			                  mCodeVectors.get(winningNeuron).value(att) 
			                  + diff);
		// subtract
		else
			mCodeVectors.get(winningNeuron).setValue(att,
			                  mCodeVectors.get(winningNeuron).value(att) 
			                  - diff);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	/**
	 * This function initializes the clusters' weights.
	 * 
	 * @return The initialized clusters
	 */
	protected Instances initClusters() {
		Instances weights = new Instances(mTrainWindow, mNumCodeVect.getValue());
		Random random = new Random(randomSeed);
		double[] instValues;
		int numC = mTrainWindow.numClasses();

		for (int cl = 0, i = 0; i < mNumCodeVect.getValue(); i++, cl++) {
			instValues = new double[mTrainWindow.numAttributes()];
			
			// Class for the codevector.
			if (cl == numC)
				cl = 0;
			
			for (int j = 0; j < mTrainWindow.numAttributes(); j++)
				if (j == mTrainWindow.classIndex())
					instValues[j] = cl;
				else
					instValues[j] = random.nextGaussian();
			
			Instance inst = new DenseInstance(1, instValues);
			inst.setDataset(weights);
			weights.add(inst);
		}

		return weights;
	}

	public String getPurposeString() {
		
		return "Online LVQ.";
	}

}
