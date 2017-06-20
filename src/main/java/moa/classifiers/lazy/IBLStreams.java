package moa.classifiers.lazy;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.core.Measurement;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.EuclideanDistance;
import moa.classifiers.lazy.neighboursearch.NormalizableDistance;

import weka.core.Utils;

import xxl.core.cursors.mappers.Mapper;
import xxl.core.functions.AbstractFunction;
import xxl.core.functions.Function;

import utils.IncrementalVariance;
import utils.indexstructure.Distance;
import utils.indexstructure.DoubleFunction;
import utils.indexstructure.MTree;

import Jama.*;

/**
 * Ammar Shaker (shaker at Mathematik dot Uni dash Marburg dot de), J�rgen
 * Beringer
 * 
 */
public class IBLStreams extends AbstractClassifier {

	// GUI options
	public IntOption widthInitOption = new IntOption("InitialWidth", 'i',
			"Size of first Window for training learner.", 1000, 1,
			Integer.MAX_VALUE);

	public MultiChoiceOption PredictionStrategyOption = new MultiChoiceOption(
			"PredictionStrategy",
			's',
			"The way the target value is predicted.\n\tWModeClass ==> WeightedMode (Classification)\n\tWMedianOClass ==> WeightedMedian (OrdinalClassification)\n\tWMeanReg ==> WeightedMean (Regression)\n\tLocLinReg ==> Local Linear Regression",
			new String[] { "WModeClass", "WMedianOClass", "WMeanReg",
					"LocLinReg" }, new String[] {
					"Weighted Mode (Classification)",
					"Weighted Median (Ordinal Classification)",
					"Weighted Mean (Regression)", "Local Linear Regression" },
			0);

	public MultiChoiceOption AdaptationStrategyOption = new MultiChoiceOption(
			"AdaptationStrategy",
			'a',
			"The used adaptation Strategy.\n\tAdaptK ==> AdaptK\n\tAdaptSigma ==> AdaptSigma\n\tnone ==> none",
			new String[] { "AdaptK", "AdaptSigma", "none" },

			new String[] { "AdaptK", "AdaptSigma", "none" }, 0);

	public MultiChoiceOption WeightingSchemeKernelOption = new MultiChoiceOption(
			"WeightingMethod",
			'w',
			"The used weighting method.\n\tequal ==> equal\n\tinverseDistance ==> inverseDistance\n\tlinear ==> linear\n\tGaussianKernel ==> GaussianKernel\n\texponentialKernel ==> exponentialKernel",
			new String[] { "equal", "inverseDistance", "linear",
					"GaussianKernel", "exponentialKernel" },

			new String[] { "equal", "inverseDistance", "linear",
					"GaussianKernel", "exponentialKernel" }, 0);

	public IntOption MaxInstancebaseSizeOption = new IntOption(
			"MaxInstancebaseSize", 'm',
			"The maximum allowed size for the instancebase.", 5000, 100,
			Integer.MAX_VALUE);

	public IntOption MinKValueOption = new IntOption(
			"MinKValue",
			'j',
			"Minimum k value, recomended 2 times the number of attributes. This value is considered when the used adaptation strategy is AdaptK.",
			4, 1, Integer.MAX_VALUE);

	public IntOption InitKValueOption = new IntOption("InitialKValue", 'k',
			"Initial k value, recomended 4 times the number of attributes.",
			16, 1, Integer.MAX_VALUE);

	public IntOption MaxKValueOption = new IntOption(
			"MaxKValue",
			'l',
			"Maximum k value, recomended 10 times the number of attributes.  This value is considered when the used adaptation strategy is AdaptK.",
			40, 1, Integer.MAX_VALUE);

	public FloatOption InitSigmaOption = new FloatOption(
			"InitialSigma",
			'q',
			"Initial sigma value. This value is considered when the used adaptation strategy is AdaptSigma.",
			0.5, 0.005, 100);

	public FlagOption UseDefaultSigmaOption = new FlagOption(
			"UseDefaultSigma",
			'h',
			"In this case the default sigma is used, when a Gaussian or an exponential kernel is used."
					+ " sig=sqrt(num. dimensions)/10.");

	public FlagOption UseDefaultkOption = new FlagOption("UseDefaultK", 'g',
			"In this case the default k." + " k=(num. dimensions)* 4.");

	/* final constants static */
	private static final long serialVersionUID = 2803962004658943062L;

	private final static double EPSILON = 0.00001;

	public static final int NOMINAL = 2;

	public static final int NUMERIC = 1;

	public static boolean DEBUG = false;

	/* general configuration variables */

	public static PredictionStrategy predictionStrategy = PredictionStrategy.LinearRegression;
	public static AdaptationStrategy adaptationStrategy = AdaptationStrategy.AdaptK;
	public static WeightingSchemeKernel kernelMethod = WeightingSchemeKernel.GaussianKernel;

	public boolean ordinalClassification = false;

	public int initialWidth = 0;

	public int kValue1 = 5;

	public int kValue2 = 50;

	public int old = 10;

	public int maxSize = 5000;

	public int histlen = 100;

	public int lastHist = 20;

	protected int young = kValue1;

	protected double minProbDiff = 0.2;

	protected double significant = 4.0;

	// Z for alpha/2 which is used to calculate the confidence interval
	protected double ZalphaDiv2 = 1.645;
	// protected double ZalphaDiv2 = 1.2 ;

	protected boolean redundance = true;

	protected boolean normalize = true;

	protected boolean useVDM = true;

	// protected NormDistance distance;

	// protected NormDistance distance2;

	protected EuclideanDistance distance;

	protected EuclideanDistance distance2;

	/* choosing different ks configuration variables */

	protected int MinkValue = 3;

	protected int kValue = 16;

	protected int MaxkValue = 40;

	/* choosing different sigmas configuration variables */

	protected double maxsigmaForKernel = 2;

	protected double minimumsigmaForKernel = 0.01;

	protected double sigmaForKernel = 0.5;

	protected double sigmaDelta = 0.05;

	protected double[] differentConfigurations = null;

	protected int totalTries = 1;

	/* variables related to the current dataset */

	protected int allAttributesCount;

	protected int usedAttributeCount;

	protected int classCount;

	protected int classIndex;

	protected int[] usedAttribute;

	protected boolean isNumericClass = false;

	// minimum value of the numeric attributes
	protected double[] AttMinVal;

	// maximum value of the numeric attributes
	protected double[] AttMaxVal;

	// class distribution of the nominal attributes
	public double[][][] distribs;

	// initial instances buffer
	protected Instances instancesBuffer;

	protected Instances header;

	protected MTree<InstanceInfo> mtree;

	// number of drifts occured until now
	protected int NumOfDrifts = 0;

	/* variables related to the current running instance */

	// number of instances processed till now
	protected int currentSize;

	protected int timeIndex = 0;

	// for giving values to the time attribute
	protected int timeSequence = 1;

	protected int tempSequence = 1;

	protected int lastPos = 0;

	protected int toDel = 0;

	protected List toDelete = new ArrayList();

	protected int warning;

	protected double[] sumShortError;

	protected double[][] ShortErrorHistory;

	protected double[] ShortDistanceHistory;

	// for the curse of dimensionality
	protected double phi;

	protected double sumShortDistance;

	protected double meanShortDistance;

	protected double p, s;

	protected double pmin = Double.MAX_VALUE;

	protected double smin = Double.MAX_VALUE;

	protected boolean EvaluationMood = false;

	protected boolean full = false;

	protected boolean isClassificationEnabled = false;

	protected int lastQueryID;

	protected InstanceInfo queryInstance;

	protected IncrementalVariance incVariance = null;

	public static double averegeDistance = -Double.MAX_VALUE;
	public static double averegeDistanceCurrentInstance = 0;
	public static boolean isConflictMode = false;
	public static boolean isConflictModeWithDistance = false;
	public static double epistemic = 0;
	public static double aleatoric = 0;

	public Distance distanceXXL = new Distance() {

		public double distance(Object object1, Object object2) {
			InstanceInfo inf1 = (InstanceInfo) object1;
			InstanceInfo inf2 = (InstanceInfo) object2;
			return IBLStreams.this.distance(inf1.inst, inf2.inst);
		}
	};

	protected DoubleFunction<InstanceInfo> timeFunc = new DoubleFunction<InstanceInfo>() {
		public double invoke(InstanceInfo inf) {
			return inf.inst.value(timeIndex);
		}
	};

	protected class InstanceInfo implements Serializable {

		public Instance inst;

		public int queryID = Integer.MIN_VALUE;

		public double queryValue = 0;

		public boolean deleted = false;

		public double maxDist = Double.POSITIVE_INFINITY;

		public InstanceInfo(Instance inst) {
			this.inst = inst;
		}

		public InstanceInfo(InstanceInfo inf) {
			this.inst = inf.inst;
		}

		public InstanceInfo(int attrCount, Instances header) {
			this.inst = new DenseInstance(1.0, new double[attrCount]);
			this.inst.setDataset(header);
			if (header == null)
				System.out.println("Error null!!!");
		}

		public String toString() {
			return inst.toString();
		}
	}

	public IBLStreams() {

		this.maxSize = 5000;
		this.old = this.kValue2 / 2;
		this.kValue2 = 50;
		this.kValue1 = 5;
		this.young = 5;
		this.minProbDiff = 0.25;
		this.significant = 4; // 1.96;
		this.histlen = 100;
		this.lastHist = 10;
		this.redundance = true;
	}

	public IBLStreams(int maxSize) {
		this.maxSize = maxSize;
	}

	public IBLStreams(int maxSize, int kValue2, int old, int young,
			double minProbDiff, double significant, int histlen, int lastHist,
			boolean red) {

		this.maxSize = maxSize;
		this.old = old;
		this.kValue2 = kValue2;
		this.young = young;
		this.minProbDiff = minProbDiff;
		this.significant = significant;
		this.histlen = histlen;
		this.lastHist = lastHist;
		this.redundance = red;
	}

	// this function is aimed to check the integrity of the chosen parameters
	protected boolean checkOptionsIntegity() {
		if (PredictionStrategyOption.getChosenIndex() == 3)
			predictionStrategy = PredictionStrategy.LinearRegression;
		else {
			predictionStrategy = PredictionStrategy.wKNN;
			if (PredictionStrategyOption.getChosenIndex() == 1)
				ordinalClassification = true;
			else
				ordinalClassification = false;
		}

		adaptationStrategy = AdaptationStrategy.values()[AdaptationStrategyOption
				.getChosenIndex()];
		kernelMethod = WeightingSchemeKernel.values()[WeightingSchemeKernelOption
				.getChosenIndex()];

		initialWidth = widthInitOption.getValue();

		maxSize = MaxInstancebaseSizeOption.getValue();

		MinkValue = MinKValueOption.getValue();
		MaxkValue = MaxKValueOption.getValue();

		if (UseDefaultSigmaOption.isSet())
			sigmaForKernel = Math.sqrt(usedAttributeCount) / 10;
		else
			sigmaForKernel = InitSigmaOption.getValue();

		maxsigmaForKernel = Math.sqrt(usedAttributeCount);

		if (UseDefaultkOption.isSet())
			kValue = usedAttributeCount * 4;
		else
			kValue = InitKValueOption.getValue();

		boolean result = false;
		if (!isNumericClass
				&& predictionStrategy == PredictionStrategy.LinearRegression)
			new Exception(
					"Local linear regression can only be choosen for regression problems");

		if (adaptationStrategy == AdaptationStrategy.AdaptSigma
				&& !(kernelMethod == WeightingSchemeKernel.exponentialKernel || kernelMethod == WeightingSchemeKernel.GaussianKernel))
			new Exception(
					"Different sigmas adaptation strategy can be used only with Gaussian or exponential kernels");

		if (MinkValue > kValue || MaxkValue < kValue)
			new Exception(
					"The min, max, initial k values are not correctly set.");

		return true;
	}

	// in this function the instance base is initialized
	public void buildClassifier(Instances data) {

		if (data.classAttribute().isNominal())
			isNumericClass = false;
		else if (data.classAttribute().isNumeric())
			isNumericClass = true;
		else
			isNumericClass = false;

		data = AddTimeIndex(data);

		header = new Instances(data, 0);
		allAttributesCount = data.numAttributes();
		classIndex = data.classIndex();
		classCount = data.numClasses();

		computeUsedAttributes();

		usedAttributeCount = 0;
		for (int i = 0; i < usedAttribute.length; i++) {
			if (usedAttribute[i] == NOMINAL || usedAttribute[i] == NUMERIC)
				usedAttributeCount++;
		}
		phi = 0.1414 * Math.sqrt(usedAttributeCount / 2);

		checkOptionsIntegity();

		AttMinVal = new double[allAttributesCount];
		AttMaxVal = new double[allAttributesCount];

		distribs = new double[allAttributesCount][][];

		// initialize fields
		for (int i = 0; i < allAttributesCount; i++) {
			if (usedAttribute[i] == NUMERIC) {
				AttMinVal[i] = Double.MAX_VALUE;
				AttMaxVal[i] = -Double.MAX_VALUE;
			} else if (usedAttribute[i] == NOMINAL) {
				distribs[i] = new double[data.attribute(i).numValues()][data
						.numClasses()];
			}
		}
		// distance = new NormDistance(data, usedAttribute, normalize, useVDM);
		// distance2 = (NormDistance) distance.clone();
		try {
			distance = new EuclideanDistance(data);
			distance.getRanges();
			distance2 = new EuclideanDistance(data);
			distance2.getRanges();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if (mtree == null) {
			mtree = new MTree(distanceXXL, 6, 15);
		} else {
			mtree.clear();
		}

		resetLearningImpl();
		toDelete.clear();

		// setting up the different Ks or sigmas configurations
		if (adaptationStrategy == AdaptationStrategy.AdaptK) {
			totalTries = 3;
			ShortErrorHistory = new double[totalTries][histlen];
			sumShortError = new double[totalTries];
			differentConfigurations = new double[totalTries];

			differentConfigurations[0] = -1;
			differentConfigurations[1] = 0;
			differentConfigurations[2] = 1;
		} else if ((kernelMethod == WeightingSchemeKernel.GaussianKernel || kernelMethod == WeightingSchemeKernel.exponentialKernel)
				&&

				adaptationStrategy == AdaptationStrategy.AdaptSigma) {
			totalTries = 3;
			ShortErrorHistory = new double[totalTries][histlen];
			sumShortError = new double[totalTries];
			differentConfigurations = new double[totalTries];

			differentConfigurations[0] = -sigmaDelta;
			differentConfigurations[1] = 0;
			differentConfigurations[2] = sigmaDelta;

		} else {
			totalTries = 1;
			ShortErrorHistory = new double[totalTries][histlen];
			sumShortError = new double[totalTries];
			differentConfigurations = new double[totalTries];

			differentConfigurations[0] = 0;
		}

		ShortDistanceHistory = new double[histlen];
		full = false;

		currentSize = 0;
		lastPos = 0;
		sumShortDistance = 0;
		meanShortDistance = 0;

		lastQueryID = Integer.MIN_VALUE;
		queryInstance = new InstanceInfo((Instance) null);
		// Enumeration enumer = data.enumerateInstances();
		//
		// while (enumer.hasMoreElements()) {
		// Instance inst = (Instance) enumer.nextElement();
		// Instance cp = (Instance) inst.copy();
		// cp.setDataset(header);
		// insert(new InstanceInfo(cp));
		// }
		for (int i = 0; i < data.numInstances(); i++) {
			Instance inst = data.instance(i);
			Instance cp = (Instance) inst.copy();
			cp.setDataset(header);
			insert(new InstanceInfo(cp));
		}

	}

	public boolean isSignificant() {
		boolean value = false;
		if (!isNumericClass)
			value = full && p + s > pmin + significant * smin
					&& p >= this.minProbDiff;
		else
			value = full && (p + s > pmin + significant * smin);

		return value;
	}

	public int getSinceWarning() {
		return warning;
	}

	public double getProbDiff() {
		int len = lastHist;
		int newLen = Math.max(0, len);

		double sumNew = 0;
		for (int i = 0; i < newLen; i++) {

			if (totalTries == 3)
				sumNew += ShortErrorHistory[1][(lastPos - i + histlen)
						% histlen];

			else if (totalTries == 1)
				sumNew += ShortErrorHistory[0][(lastPos - i + histlen)
						% histlen];
		}
		return (sumNew / (double) (newLen)) - (pmin);
	}

	@Override
	public void resetLearningImpl() {
		pmin = Double.MAX_VALUE;
		smin = Double.MAX_VALUE;

		// classifyCount = (int) (histlen * 0.5);

		lastPos = 1;
		full = false;
		warning = 0;
	}

	public void insert(InstanceInfo inst) {
		if (inst == null)
			System.out.println("ERROR null inserted");

		addInstanceStatisitcs(inst.inst);
		mtree.add(inst);
		currentSize++;
	}

	public void remove(InstanceInfo inst) {

		if (mtree.remove(inst)) {
			currentSize--;
			// distance2.removeInstance(inst.inst);
			removeInstance(distance2, inst.inst);
			if (mtree.remove(inst)) {
				System.out.println("�hm...");
				mtree.remove(inst);
			}
		}
	}

	/**
	 * Remove an instance from a distance function. Added by Álvar
	 * Arnaiz-González.
	 * 
	 * @param distance
	 *            Distance function from which it will be removed the instance.
	 * @param inst
	 *            Instance to remove.
	 */
	private void removeInstance(NormalizableDistance distance, Instance inst) {
		Instances oldInsts = distance.getInstances(), newInsts = new Instances(
				oldInsts, oldInsts.numInstances() - 1);

		for (int i = 0; i < oldInsts.numInstances(); i++) {
			if (!oldInsts.instance(i).toString().equals(inst.toString()))
				newInsts.add(oldInsts.instance(i));
		}

		distance.setInstances(newInsts);
	}

	/**
	 * Remove an iterator of instances from a distance function. Added by Álvar
	 * Arnaiz-González.
	 * 
	 * @param distance
	 *            Distance function from which it will be removed the instance.
	 * @param it
	 *            Iterator with the instances to remove.
	 */
	private void removeInstances(NormalizableDistance distance,
			Iterator<Instance> it) {
		Instances oldInsts = distance.getInstances(), newInsts = new Instances(
				oldInsts, oldInsts.numInstances() - 1);
		ArrayList<Instance> toRemove = new ArrayList<>();
		boolean add;

		// Store the instances to remove.
		while (it.hasNext())
			toRemove.add(it.next());

		// Not add to newInsts the instances that exist in toRemove.
		for (int i = 0; i < oldInsts.numInstances(); i++) {
			add = true;

			for (int j = 0; j < toRemove.size(); j++) {
				if (!oldInsts.instance(i).toString()
						.equals(toRemove.get(j).toString())) {
					add = false;
					toRemove.remove(j);
					break;
				}
			}

			if (add)
				newInsts.add(oldInsts.instance(i));
		}

		distance.setInstances(newInsts);
	}

	/*
	 * updating the sumShortError array which holds the occurred errors in the
	 * last histlen instances, for the current configurations, candidate
	 * configurations Updates also the pmin and smin
	 */
	public void UpateClassificationErrorRate(double[] error) {

		for (int i = 0; i < error.length; i++) {
			sumShortError[i] -= ShortErrorHistory[i][lastPos];
			sumShortError[i] += error[i];

			ShortErrorHistory[i][lastPos] = error[i];
		}

		lastPos = (lastPos + 1) % histlen;
		if ((lastPos == histlen - 1) && !full) {
			// classifyCount++;
			// if (classifyCount >= histlen)
			full = true;

			if (totalTries == 3)
				incVariance = new IncrementalVariance(ShortErrorHistory[1]);
			else if (totalTries == 1)
				incVariance = new IncrementalVariance(ShortErrorHistory[0]);
		}
		if (!full)
			return;

		if (isNumericClass || ordinalClassification) {
			if (totalTries == 3)
				incVariance.UpdateMeanVarianceOnWindow(error[1]);
			else if (totalTries == 1)
				incVariance.UpdateMeanVarianceOnWindow(error[0]);

			p = incVariance.Mean;
			s = Math.sqrt(incVariance.Variance);
		} else {
			if (totalTries == 3)
				p = sumShortError[1] / histlen;

			else if (totalTries == 1)
				p = sumShortError[0] / histlen;

			s = Math.sqrt(p * (1 - p) / histlen);
		}

		if (p + s < pmin + smin) {
			pmin = p;
			smin = s;
			warning = 0;
		} else {
			if (p + s > pmin + 1.96 * smin
					&& (isNumericClass || (p > this.minProbDiff / 2))) {
				warning++;
			} else
				warning = 0;
		}
	}

	@Override
	public void trainOnInstanceImpl(Instance newInst) {
		tempSequence++;
		
		if (widthInitOption.getValue() != 0 && !isClassificationEnabled) {

			if (instancesBuffer == null) {
				// this.instancesBuffer = new Instances(newInst.dataset());
				this.instancesBuffer = new Instances(newInst.dataset(), 0);
			}

			instancesBuffer.add(newInst);

			if (instancesBuffer.size() == widthInitOption.getValue()) {
				// Build first time Classifier
				buildClassifier(instancesBuffer);
				isClassificationEnabled = true;
			}

			return;
		}

		Instance inst = (Instance) newInst.copy();

		inst.setDataset(header);

		inst = AddTimeIndex(inst, false);

		// distance2.addInstance(inst);
		distance2.update(inst);

		updateInstanceBase(inst);

		// here is the checking whether we have a concept change
		if (inst.classAttribute().isNominal()) {

			// if ((1 - p) <= 1.0 / classCount
			// ||(isSignificant() && getSinceWarning() >= lastHist - 2)) {
			if (isSignificant()) {
				double pDiff = getProbDiff();

				if ((p - pmin) > minProbDiff && pDiff > minProbDiff) {
					double oldpDiff = pDiff;
					if ((1 - p) <= 1.0 / classCount
							|| ((1 - pmin) - pDiff) <= 1.0 / classCount)
						pDiff = 1.0;
					int toDel = Math.min((int) (currentSize * pDiff),
							currentSize - kValue2);
					// System.out.println("toDel\t" + toDel+"\tcount\t"
					// +currentSize) ;
					if (toDel > 0) {
						NumOfDrifts++;
						doDelete();
						List removed = mtree.remove(toDel, timeFunc);
						int rem = removed.size();
						AbstractFunction map = new AbstractFunction<InstanceInfo, Instance>() {
							public Instance invoke(InstanceInfo ob) {
								return ob.inst;
							}
						};
						Function o;
						Function<InstanceInfo, Instance> g;
						// distance2.removeInstances(new Mapper(map,
						// removed.iterator()));
						removeInstances(distance2,
								new Mapper(map, removed.iterator()));
						currentSize -= rem;
						if (DEBUG)
							System.out
									.println("STATS: " + rem + "    " + pDiff
											+ "  " + oldpDiff + "  " + p
											+ "   " + pmin);
						resetLearningImpl();
					}
				}
			}
		} else if (inst.classAttribute().isNumeric()) {
			// if (isSignificant() && getSinceWarning() >= lastHist - 2) {
			if (isSignificant()) {
				double pDiff = getProbDiff();

				// if ((p - pmin) > minProbDiff && pDiff > minProbDiff) {
				if (true) {
					double pp = Math.min((p - pmin) / pmin, 0.5);
					// System.out.println("(p-pmin)/pmin\t" + (p-pmin)/pmin) ;
					int toDel = Math.min((int) (currentSize * pp), currentSize
							- kValue2);
					// System.out.println("toDel\t" + toDel+"\tcount\t"
					// +currentSize) ;
					if (toDel > 0) {
						NumOfDrifts++;
						doDelete();
						List removed = mtree.remove(toDel, timeFunc);
						int rem = removed.size();
						AbstractFunction map = new AbstractFunction<InstanceInfo, Instance>() {
							public Instance invoke(InstanceInfo ob) {
								return ob.inst;
							}
						};
						// distance2.removeInstances(new Mapper(map,
						// removed.iterator()));
						removeInstances(distance2,
								new Mapper(map, removed.iterator()));
						currentSize -= rem;
						if (DEBUG)
							System.out.println("STATS: " + rem + "    " + pDiff
									+ "  " + pp + "  " + p + "   " + pmin);
						resetLearningImpl();
					}
				}
			}
		}

		// here we check wehtehr a different sigma should be used
		// in the case where Gussian or exponential kernels are used
		if (full && (totalTries != 1)) {
			double currentP = 0;
			double currentS = 0;

			if (differentConfigurations.length == 3)
				currentP = sumShortError[1] / histlen;
			else if (differentConfigurations.length == 1)
				currentP = sumShortError[0] / histlen;

			for (int i = 0; i < sumShortError.length; i++) {
				if (adaptationStrategy == AdaptationStrategy.AdaptK && i == 1)
					continue;
				else if (adaptationStrategy == AdaptationStrategy.AdaptSigma
						&& i == 1)
					continue;

				else if (adaptationStrategy == AdaptationStrategy.AdaptSigma
						&& (sigmaForKernel * (1 + differentConfigurations[i]) <= minimumsigmaForKernel || sigmaForKernel
								* (1 + differentConfigurations[i]) >= maxsigmaForKernel))
					continue;

				double tempP = sumShortError[i] / histlen;

				if ((currentP > tempP)) {
					if (adaptationStrategy == AdaptationStrategy.AdaptK) {
						if ((kValue + differentConfigurations[i] != MinkValue)
								&& (kValue + differentConfigurations[i] != MaxkValue)) {
							kValue += differentConfigurations[i];
							// System.out.println("new k: " + kValue );
						}
						// else
						// System.out.println("new k cannot occure: " + kValue+
						// differentConfigurations[i]);
					} else if (adaptationStrategy == AdaptationStrategy.AdaptSigma) {
						// System.out.print("old Sigma: " + sigmaForKernel
						// +" \t ");
						sigmaForKernel *= (1 + differentConfigurations[i]);
						// System.out.println("new Sigma: " + sigmaForKernel );
					}

					resetLearningImpl();
					break;
				}
			}
		}
	}

	/*
     * 
     */
	protected void updateInstanceBase(Instance inst) {
		double[] cls = null;
		try {
			cls = classifyInstanceDifferentSettings(inst);

			double[] tempErrors = new double[totalTries];
			boolean consider = true;
			for (int i = 0; i < totalTries; i++) {
				if (Double.isNaN(cls[i])) {
					consider = false;
					break;
				}
				if (inst.classAttribute().isNominal()) {
					if (ordinalClassification)
						tempErrors[i] = Math.abs(cls[i]
								- (int) inst.classValue());
					else
						tempErrors[i] = (cls[i] == (int) inst.classValue() ? 0
								: 1);
				} else if (inst.classAttribute().isNumeric()) {
					tempErrors[i] = Math.abs(cls[i] - inst.classValue());
				} else {
					tempErrors[i] = (cls[i] == (int) inst.classValue() ? 0 : 1);
				}
			}
			if (consider)
				UpateClassificationErrorRate(tempErrors);
		} catch (Exception e) {
			e.printStackTrace();
		}

		doDelete();

		updateInceanceNeighborhood(inst);

		doDelete();
		insert(new InstanceInfo(inst));

		if (currentSize > maxSize) {
			if (currentSize - maxSize > kValue1) {
				List c = mtree.remove(currentSize - maxSize, timeFunc);
				currentSize -= c.size();
			}
			toDel = currentSize - maxSize;
		}

	}

	/**
	 * Insert Instance in the sorted list for search of nearest instances.
	 * 
	 * The instance to be inserted is discarded in case the list has at least
	 * maxsize instances and the instance query value is larger than that of
	 * last instance
	 * 
	 * @param list
	 * @param inf
	 * @param maxSize
	 * @return
	 */
	protected double insert(List list, InstanceInfo inf, int maxSize) {
		double val = inf.queryValue;
		double max = Double.MAX_VALUE;
		if (list.size() >= maxSize) {
			max = ((InstanceInfo) list.get(maxSize - 1)).queryValue;
		}
		if (max < val)
			return max;
		int p = 0;
		while (p < list.size()
				&& ((InstanceInfo) list.get(p)).queryValue <= val) {
			p++;
		}
		list.add(p, inf);
		if (list.size() >= maxSize) {
			max = ((InstanceInfo) list.get(maxSize - 1)).queryValue;
			while (((InstanceInfo) list.get(list.size() - 1)).queryValue > max) {
				list.remove(list.size() - 1);
			}
		}
		return max;
	}

	protected void updateInceanceNeighborhood(Instance inst) {

		double instcls = inst.classValue();

		queryInstance.inst = inst;
		lastQueryID++;
		queryInstance.queryID = lastQueryID;
		// queryComparator.queryID = lastQueryID;
		// mtree.checkRadius();
		Iterator nearest = mtree.getNearestNeighbours(queryInstance);
		InstanceInfo[] near = new InstanceInfo[kValue2];
		double time[] = new double[kValue2];
		double cls[] = new double[kValue2];
		for (int i = 0; i < kValue2 && nearest.hasNext(); i++) {
			InstanceInfo inf = (InstanceInfo) nearest.next();
			if (inf == null)
				return;
			if (inf.deleted) {
				System.out.println("DEL");
			}
			if (inf.queryID != lastQueryID) {
				inf.queryValue = distance(inst, inf.inst);
				inf.queryID = lastQueryID;
			}
			near[i] = inf;
			time[i] = inf.inst.value(timeIndex);
			cls[i] = inf.inst.classValue();

		}
		if (near[kValue2 - 1] == null)
			return;

		sumShortDistance -= ShortDistanceHistory[lastPos];
		ShortDistanceHistory[lastPos] = distance.distance(inst,
				near[kValue1 - 1].inst);
		sumShortDistance += ShortDistanceHistory[lastPos];
		meanShortDistance = sumShortDistance / histlen;

		boolean del = false;
		do {
			del = false;
			int[] timesort = Utils.sort(time);

			if (near[kValue2 - 1] != null)
				del = forgetContradictingNeighbors(inst, instcls, near, cls,
						time, timesort);

			if (del) {
				// System.out.println("ERROR");
				for (int i = 0; i < kValue2; i++) {
					if (near[i] == null)
						break;
					if (near[i].deleted) {
						System.arraycopy(near, i + 1, near, i, kValue2 - i - 1);
						System.arraycopy(time, i + 1, time, i, kValue2 - i - 1);
						System.arraycopy(cls, i + 1, cls, i, kValue2 - i - 1);
						if (nearest.hasNext()) {
							InstanceInfo inf = (InstanceInfo) nearest.next();
							if (inf.queryID != lastQueryID) {
								inf.queryValue = distance(inst, inf.inst);
								inf.queryID = lastQueryID;
							}
							near[kValue2 - 1] = inf;
							time[kValue2 - 1] = inf.inst.value(timeIndex);
							cls[kValue2 - 1] = inf.inst.classValue();
						} else
							// shouldn't this be?
							// near[kValue2 -i - 1] = null;
							near[kValue2 - 1] = null;
						i--;
					}
				}
			}
		} while (del);
	}

	// timesort is the ascending sort of time array (containing only indexes)
	protected boolean forgetContradictingNeighbors(Instance newInst, double cl,
			InstanceInfo[] kInst, double[] kcls, double[] time, int[] timesort) {

		boolean isDeleted = false;

		ConfidenceInterval cInterval = null;

		int[] classYoungInstancesHistogram = null;

		double[] neighborInstances = new double[kValue1];

		int countYoungInstancesBelongToInterval = 0;
		int countInstancesInLargeNeighborWithCl = 0;

		// countInstancesInLargeNeighborWithCl represents how many instances
		// only in the
		// long buffer kValue2 having the same class value as the current
		// instance

		// countYoung holds the number of close instances to the current
		// instance which
		// have the same class value

		double youngT = time[timesort[kValue2 - young]];
		// finding the youngest nearest instance

		int ca = 0;
		int countYoung = kValue1 + 1;
		if (isNumericClass) {
			for (int i = 1; i <= kValue1; i++) {
				neighborInstances[i - 1] = kcls[timesort[kValue2 - i]];
			}
			cInterval = new ConfidenceInterval(neighborInstances, ZalphaDiv2);
			if (cInterval.belongToInterval(cl))
				countYoungInstancesBelongToInterval++;
			for (int i = 1; i <= young; i++) {
				if (cInterval.belongToInterval(kcls[timesort[kValue2 - i]]))
					countYoungInstancesBelongToInterval++;
			}

			ca = countYoung;
			for (int i = young + 1; i <= kValue2; i++) {
				ca++;
				if (cInterval.belongToInterval(kcls[timesort[kValue2 - i]]))
					countInstancesInLargeNeighborWithCl++;
			}
		} else {
			classYoungInstancesHistogram = new int[classCount];
			classYoungInstancesHistogram[(int) cl]++;
			for (int i = 1; i <= young; i++) {
				classYoungInstancesHistogram[(int) kcls[timesort[kValue2 - i]]]++;
			}
			countInstancesInLargeNeighborWithCl = classYoungInstancesHistogram[(int) cl];
			ca = countYoung;
			for (int i = young + 1; i <= kValue2; i++) {
				ca++;
				if (kcls[timesort[kValue2 - i]] == cl) {
					countInstancesInLargeNeighborWithCl++;
				}
			}
		}

		int oldestIdInTheNearest = -1;
		double oldestInTheNearest = Double.MAX_VALUE;

		// finding the oldest instance between the near neighbors
		for (int i = 1; i <= kValue1; i++) {
			if (time[i - 1] < oldestInTheNearest) {
				oldestInTheNearest = time[i - 1];
				oldestIdInTheNearest = i - 1;
			}
		}

		if (isNumericClass) {

			boolean isRedundance = (countInstancesInLargeNeighborWithCl
					/ (double) ca >= 0.5 && countYoungInstancesBelongToInterval >= (countYoung / 2))
					&& (meanShortDistance > distance.distance(newInst,
							kInst[kValue1 - 1].inst));

			if (cInterval.belongToInterval(cl)) {
				// for (int i = 0; i < old; i++) {
				// shouldn't this be timesort[i]>kValue1 ??
				// if (timesort[i]<kValue1 &&
				// !cInterval.belongToInterval(kcls[timesort[i]]))
				for (int i = 0; i < Math.min(timesort.length, old); i++) {
					double dist = distance(newInst, kInst[timesort[i]].inst);

					// if (i<timesort.length-kValue1 &&
					// !cInterval.belongToInterval(kcls[timesort[i]])
					// && dist < phi )
					if (!cInterval.belongToInterval(kcls[timesort[i]])
							&& dist < phi) {
						isDeleted = true;
						toDelete.add(kInst[timesort[i]]);
						kInst[timesort[i]].deleted = true;
					}
				}
			}
			if (redundance && isRedundance) {

				for (int i = 0; i < old; i++) {
					if (!kInst[timesort[i]].deleted && timesort[i] < kValue1) {
						isDeleted = true;
						toDelete.add(kInst[timesort[i]]);
						kInst[timesort[i]].deleted = true;
					}
				}
			}
			if (!isDeleted
					&& (toDel > 0 || maxSize <= currentSize - toDelete.size())) {
				// isDeleted = true;
				toDelete.add(kInst[oldestIdInTheNearest]);
				kInst[oldestIdInTheNearest].deleted = true;
				if (toDel > 0)
					toDel--;
			}
		} else {
			boolean isRedundance = (countInstancesInLargeNeighborWithCl
					/ (double) ca >= 0.98 && classYoungInstancesHistogram[(int) cl] == countYoung)
					&& (meanShortDistance > distance.distance(newInst,
							kInst[kValue1 - 1].inst));

			int[] sortedYounger = Utils.sort(classYoungInstancesHistogram);

			if (sortedYounger[classCount - 1] == cl
					&& (classYoungInstancesHistogram[(int) cl] - classYoungInstancesHistogram[sortedYounger[classCount - 2]])
							/ ((double) countYoung) > 1.0 / (classCount + 1)) {
				for (int i = 0; i < old; i++) {
					// shouldn't this be timesort[i]>kValue1 ??
					if (timesort[i] < kValue1 && (kcls[timesort[i]] != cl)) {
						isDeleted = true;
						toDelete.add(kInst[timesort[i]]);
						kInst[timesort[i]].deleted = true;
					}
				}
			}

			if (redundance && isRedundance) {

				for (int i = 0; i < old; i++) {
					if (!kInst[timesort[i]].deleted && timesort[i] < kValue1) {
						isDeleted = true;
						toDelete.add(kInst[timesort[i]]);
						kInst[timesort[i]].deleted = true;
					}
				}
			}
			if (!isDeleted
					&& (toDel > 0 || maxSize <= currentSize - toDelete.size())) {
				// isDeleted = true;
				toDelete.add(kInst[oldestIdInTheNearest]);
				kInst[oldestIdInTheNearest].deleted = true;
				if (toDel > 0)
					toDel--;
			}
		}
		return isDeleted;
	}

	protected void doDelete() {
		if (toDelete.size() > 0) {
			if (DEBUG)
				System.out.print(toDelete.size() + " Instanzen gel�scht.     ");
		} else
			return;
		Iterator it = toDelete.iterator();
		while (it.hasNext()) {
			InstanceInfo inf = (InstanceInfo) it.next();
			remove(inf);
		}
		// hasDeleted(mtree);
		toDelete.clear();

	}

	/*
	 * this function returns the class value for different possible ks or sigmas
	 */
	public double[] classifyInstanceDifferentSettings(Instance instance)
			throws Exception {
		// System.out.println("classifyInstance") ;
		double cls = instance.classValue();

		double[][] dist = null; // getPredictionDistribution(instance);

		if (adaptationStrategy == AdaptationStrategy.AdaptK)
			dist = getPredictionDistributionKs(instance);
		else
			dist = getPredictionDistributionKernel(instance);

		double[] result = new double[dist.length];

		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}

		for (int k = 0; k < dist.length; k++) {
			if (dist[k] == null) {
				result[k] = Double.NaN;
				continue;
			}

			if (instance.classAttribute().isNominal()) {
				if (!ordinalClassification) {
					int maxIndex = getIndexofMaxElement(dist[k]);
					result[k] = maxIndex;
				} else {
					int medIndex = getIndexofMedianElement(dist[k]);
					result[k] = medIndex;
				}
				break;
			} else if (instance.classAttribute().isNumeric()) {
				// System.out.println("Cl  "+instance.toString() + "  " +
				// dist[0]) ;
				result[k] = dist[k][0];
				break;
			} else {
				// System.out.println("Cl  "+instance.toString() + "  " +
				// Instance.missingValue()) ;
				result[k] = Double.NaN;
				// result[k] = Instance.missingValue();
			}
		}
		return result;
	}

	/*
	 * this function returns the class value
	 */
	public double classifyInstance(Instance instance) throws Exception {
		// System.out.println("classifyInstance") ;
		double cls = instance.classValue();

		double[] dist = null;

		if (adaptationStrategy == AdaptationStrategy.AdaptK)
			dist = getPredictionDistributionKs(instance)[kValue];
		else if (totalTries == 3)
			dist = getPredictionDistributionKernel(instance)[1];
		else if (totalTries == 1)
			dist = getPredictionDistributionKernel(instance)[0];

		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}

		if (instance.classAttribute().isNominal()) {
			double max = 0;
			int maxIndex = 0;
			for (int i = 0; i < dist.length; i++) {
				if (dist[i] > max) {
					maxIndex = i;
					max = dist[i];
				}
			}

			if (max > 0) {
				return maxIndex;
			} else {
				return Double.NaN;
				// return Instance.missingValue();
			}
		} else if (instance.classAttribute().isNumeric()) {
			return dist[0];
		} else {
			return Double.NaN;
			// return Instance.missingValue();
		}
	}

	/**
	 * Predicts the class memberships for a given instance. If an instance is
	 * unclassified, the returned array elements must be all zero.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return an array containing the estimated membership probabilities of the
	 *         test instance in each class
	 * @exception Exception
	 *                if distribution could not be computed successfully
	 */
	@Override
	public double[] getVotesForInstance(Instance instance) {

		if (mtree == null) {
			double[] dist = new double[1];
			dist[0] = 1;
			return dist;
		}
		Instance instanceC = AddTimeIndex((Instance) instance.copy(), true);
		addInstanceStatisitcs(instanceC);
		EvaluationMood = true;
		double[] dob = null;

		if (adaptationStrategy == AdaptationStrategy.AdaptK)
			dob = getPredictionDistributionKs(instanceC)[1];
		else if (totalTries == 3)
			dob = getPredictionDistributionKernel(instance)[1];

		else if (totalTries == 1)
			dob = getPredictionDistributionKernel(instance)[0];

		if (!isNumericClass && ordinalClassification) {
			int medIndex = getIndexofMedianElement(dob);
			for (int i = 0; i < dob.length; i++)
				dob[i] = 0;
			dob[medIndex] = 1;
		}

		EvaluationMood = false;
		// System.out.println(instance + " " + dob[0] + " " + dob[1] ) ;

		double pPlus = 0;
		double pMinus = 0;

		if (isConflictMode) {
			if (isConflictModeWithDistance) {
				dob[0] = (this.averegeDistanceCurrentInstance / this.averegeDistance)
						* dob[0];
				dob[1] = (this.averegeDistanceCurrentInstance / this.averegeDistance)
						* dob[1];
			}

			epistemic = Math.min(1 - dob[0], 1 - dob[1]);
			aleatoric = Math.min(dob[0], dob[1]);
			if (dob[0] > dob[1]) {
				pPlus = dob[0] - dob[1];
				pMinus = 0;
			} else {
				pPlus = 0;
				pMinus = dob[1] - dob[0];
			}
			dob[0] = pPlus;
			dob[1] = pMinus;
		}
		return dob;
	}

	protected double[][] getWeightsFromDistances(double[] arrayDistance) {
		double tempAverage = avergeVector(arrayDistance);

		if (averegeDistance == -Double.MAX_VALUE)
			averegeDistance = tempAverage;

		if (!EvaluationMood && tempAverage > averegeDistance)
			averegeDistance = tempAverage;
		else
			averegeDistanceCurrentInstance = Math.min(1, tempAverage
					/ averegeDistance);

		// 27.12.2011
		arrayDistance = NormalizeVector(arrayDistance);

		double[][] weightKernel = new double[totalTries][];

		double[] tempArrayDistance;
		switch (adaptationStrategy) {
		case AdaptK:
			tempArrayDistance = new double[arrayDistance.length - 1];

			for (int i = 0; i < tempArrayDistance.length; i++)
				tempArrayDistance[i] = arrayDistance[i];

			break;
		default:
			tempArrayDistance = arrayDistance;
		}

		double[] tempweightKernel = getWeightVectorKernel(tempArrayDistance,
				kernelMethod, sigmaForKernel);

		boolean fixedSigma = false;

		double epsilon = 0.005;

		double sum = sumArrayElements(tempweightKernel);
		double temp = 0;

		if ((predictionStrategy == PredictionStrategy.LinearRegression)
				&& (kernelMethod == WeightingSchemeKernel.exponentialKernel || kernelMethod == WeightingSchemeKernel.GaussianKernel)
				&&

				tempweightKernel[tempweightKernel.length / 2] / sum < epsilon) {
			fixedSigma = true;

			switch (kernelMethod) {
			case GaussianKernel: {
				temp = Math.sqrt((-2 * Math.log(sum * epsilon)));
				temp = arrayDistance[(tempweightKernel.length + 1) / 2]
						/ (temp);
				tempweightKernel = getWeightVectorKernel(arrayDistance,
						kernelMethod, temp);
				break;
			}
			case exponentialKernel: {
				temp = (-2 * Math.log(sum * epsilon));
				temp = arrayDistance[(tempweightKernel.length + 1) / 2]
						/ (temp);
				temp = Math.sqrt(temp);
				tempweightKernel = getWeightVectorKernel(arrayDistance,
						kernelMethod, temp);
				break;
			}
			}
			if (totalTries == 3)
				weightKernel[1] = NormalizeVector(tempweightKernel);
			else if (totalTries == 1)
				weightKernel[0] = NormalizeVector(tempweightKernel);

			// System.out.println("Sigma switch:\t"+sigmaForKernel+"\t"+ temp) ;
		} else if ((totalTries == 3)
				&& (predictionStrategy == PredictionStrategy.LinearRegression)
				&& (kernelMethod == WeightingSchemeKernel.exponentialKernel || kernelMethod == WeightingSchemeKernel.GaussianKernel)) {
			double[] tempArrayDistance1 = null;
			double[] tempArrayDistance2 = null;

			double[] tempweightKernel1 = null;
			double[] tempweightKernel2 = null;
			double sum1 = 0;
			double sum2 = 0;

			switch (adaptationStrategy) {
			case AdaptK:
				tempArrayDistance1 = new double[arrayDistance.length - 2];
				tempArrayDistance2 = new double[arrayDistance.length];

				for (int i = 0; i < tempArrayDistance1.length; i++) {
					tempArrayDistance1[i] = arrayDistance[i];
					tempArrayDistance2[i] = arrayDistance[i];
				}

				tempArrayDistance2[arrayDistance.length - 2] = arrayDistance[arrayDistance.length - 2];
				tempArrayDistance2[arrayDistance.length - 1] = arrayDistance[arrayDistance.length - 1];

				tempweightKernel1 = getWeightVectorKernel(tempArrayDistance1,
						kernelMethod, sigmaForKernel);
				sum1 = sumArrayElements(tempweightKernel1);

				tempweightKernel2 = getWeightVectorKernel(tempArrayDistance2,
						kernelMethod, sigmaForKernel);

				sum2 = sumArrayElements(tempweightKernel2);

				break;
			default:
				tempArrayDistance1 = arrayDistance;
				tempArrayDistance2 = arrayDistance;

				if ((sigmaForKernel * (1 + differentConfigurations[0])) > 0) {
					tempweightKernel1 = getWeightVectorKernel(
							tempArrayDistance1, kernelMethod, sigmaForKernel
									* (1 + differentConfigurations[0]));
					sum1 = sumArrayElements(tempweightKernel1);
				}

				tempweightKernel2 = getWeightVectorKernel(tempArrayDistance2,
						kernelMethod, sigmaForKernel
								* (1 + differentConfigurations[2]));

				sum2 = sumArrayElements(tempweightKernel2);
			}
			if (((sigmaForKernel * (1 + differentConfigurations[0])) > 0 || adaptationStrategy == AdaptationStrategy.AdaptK)
					&& tempweightKernel1[tempArrayDistance1.length / 2] / sum1 > epsilon) {
				weightKernel[0] = NormalizeVector(tempweightKernel1);
				weightKernel[1] = NormalizeVector(tempweightKernel);
				weightKernel[2] = NormalizeVector(tempweightKernel2);
			} else if (tempweightKernel2[tempArrayDistance2.length / 2] / sum2 > epsilon) {
				weightKernel[1] = NormalizeVector(tempweightKernel);
				weightKernel[2] = NormalizeVector(tempweightKernel2);
			} else {
				weightKernel[1] = NormalizeVector(tempweightKernel);
			}
		} else if (totalTries == 3) {
			double[] tempArrayDistance1 = null;
			double[] tempArrayDistance2 = null;

			double[] tempweightKernel1 = null;
			double[] tempweightKernel2 = null;

			switch (adaptationStrategy) {
			case AdaptK:
				tempArrayDistance1 = new double[arrayDistance.length - 2];
				tempArrayDistance2 = new double[arrayDistance.length];

				for (int i = 0; i < tempArrayDistance1.length; i++) {
					tempArrayDistance1[i] = arrayDistance[i];
					tempArrayDistance2[i] = arrayDistance[i];
				}

				tempArrayDistance2[arrayDistance.length - 2] = arrayDistance[arrayDistance.length - 2];
				tempArrayDistance2[arrayDistance.length - 1] = arrayDistance[arrayDistance.length - 1];

				tempweightKernel1 = getWeightVectorKernel(tempArrayDistance1,
						kernelMethod, sigmaForKernel);

				tempweightKernel2 = getWeightVectorKernel(tempArrayDistance2,
						kernelMethod, sigmaForKernel);

				break;
			default:
				tempArrayDistance1 = arrayDistance;
				tempArrayDistance2 = arrayDistance;

				tempweightKernel1 = getWeightVectorKernel(tempArrayDistance1,
						kernelMethod, sigmaForKernel
								* (1 + differentConfigurations[0]));

				tempweightKernel2 = getWeightVectorKernel(tempArrayDistance2,
						kernelMethod, sigmaForKernel
								* (1 + differentConfigurations[2]));
			}
			weightKernel[0] = NormalizeVector(tempweightKernel1);
			weightKernel[1] = NormalizeVector(tempweightKernel);
			weightKernel[2] = NormalizeVector(tempweightKernel2);
		} else if (totalTries == 1)
			weightKernel[0] = NormalizeVector(tempweightKernel);

		return weightKernel;
	}

	// this weighting function is used for testing purposes
	// it was created mainly to evaluate both conflict and ignorance concepts
	protected double[][] geNonNormalisedWeightsFromDistances(
			double[] arrayDistance) {
		double tempAverage = avergeVector(arrayDistance);

		// 27.12.2011
		arrayDistance = DivideVectorbyLargest(arrayDistance);

		double[][] weightKernel = new double[totalTries][];

		double[] tempArrayDistance;
		switch (adaptationStrategy) {
		case AdaptK:
			tempArrayDistance = new double[arrayDistance.length - 1];

			for (int i = 0; i < tempArrayDistance.length; i++)
				tempArrayDistance[i] = arrayDistance[i];

			break;
		default:
			tempArrayDistance = arrayDistance;
		}

		double[] tempweightKernel = getWeightVectorKernel(tempArrayDistance,
				kernelMethod, sigmaForKernel);

		boolean fixedSigma = false;

		double epsilon = 0.005;

		double sum = sumArrayElements(tempweightKernel);
		double temp = 0;

		if ((predictionStrategy == PredictionStrategy.LinearRegression)
				&& (kernelMethod == WeightingSchemeKernel.exponentialKernel || kernelMethod == WeightingSchemeKernel.GaussianKernel)
				&&

				tempweightKernel[tempweightKernel.length / 2] / sum < epsilon) {
			fixedSigma = true;

			switch (kernelMethod) {
			case GaussianKernel: {
				temp = Math.sqrt((-2 * Math.log(sum * epsilon)));
				temp = arrayDistance[(tempweightKernel.length + 1) / 2]
						/ (temp);
				tempweightKernel = getWeightVectorKernel(arrayDistance,
						kernelMethod, temp);
				break;
			}
			case exponentialKernel: {
				temp = (-2 * Math.log(sum * epsilon));
				temp = arrayDistance[(tempweightKernel.length + 1) / 2]
						/ (temp);
				temp = Math.sqrt(temp);
				tempweightKernel = getWeightVectorKernel(arrayDistance,
						kernelMethod, temp);
				break;
			}
			}
			if (totalTries == 3)
				weightKernel[1] = DivideVectorbyLargestandLength(tempweightKernel);
			else if (totalTries == 1)
				weightKernel[0] = DivideVectorbyLargestandLength(tempweightKernel);

			// System.out.println("Sigma switch:\t"+sigmaForKernel+"\t"+ temp) ;
		} else if ((totalTries == 3)
				&& (predictionStrategy == PredictionStrategy.LinearRegression)
				&& (kernelMethod == WeightingSchemeKernel.exponentialKernel || kernelMethod == WeightingSchemeKernel.GaussianKernel)) {
			double[] tempArrayDistance1 = null;
			double[] tempArrayDistance2 = null;

			double[] tempweightKernel1 = null;
			double[] tempweightKernel2 = null;
			double sum1 = 0;
			double sum2 = 0;

			switch (adaptationStrategy) {
			case AdaptK:
				tempArrayDistance1 = new double[arrayDistance.length - 2];
				tempArrayDistance2 = new double[arrayDistance.length];

				for (int i = 0; i < tempArrayDistance1.length; i++) {
					tempArrayDistance1[i] = arrayDistance[i];
					tempArrayDistance2[i] = arrayDistance[i];
				}

				tempArrayDistance2[arrayDistance.length - 2] = arrayDistance[arrayDistance.length - 2];
				tempArrayDistance2[arrayDistance.length - 1] = arrayDistance[arrayDistance.length - 1];

				tempweightKernel1 = getWeightVectorKernel(tempArrayDistance1,
						kernelMethod, sigmaForKernel);
				sum1 = sumArrayElements(tempweightKernel1);

				tempweightKernel2 = getWeightVectorKernel(tempArrayDistance2,
						kernelMethod, sigmaForKernel);

				sum2 = sumArrayElements(tempweightKernel2);

				break;
			default:
				tempArrayDistance1 = arrayDistance;
				tempArrayDistance2 = arrayDistance;

				if ((sigmaForKernel * (1 + differentConfigurations[0])) > 0) {
					tempweightKernel1 = getWeightVectorKernel(
							tempArrayDistance1, kernelMethod, sigmaForKernel
									* (1 + differentConfigurations[0]));
					sum1 = sumArrayElements(tempweightKernel1);
				}

				tempweightKernel2 = getWeightVectorKernel(tempArrayDistance2,
						kernelMethod, sigmaForKernel
								* (1 + differentConfigurations[2]));

				sum2 = sumArrayElements(tempweightKernel2);
			}
			if (((sigmaForKernel * (1 + differentConfigurations[0])) > 0 || adaptationStrategy == AdaptationStrategy.AdaptK)
					&& tempweightKernel1[tempArrayDistance1.length / 2] / sum1 > epsilon) {
				weightKernel[0] = DivideVectorbyLargestandLength(tempweightKernel1);
				weightKernel[1] = DivideVectorbyLargestandLength(tempweightKernel);
				weightKernel[2] = DivideVectorbyLargestandLength(tempweightKernel2);
			} else if (tempweightKernel2[tempArrayDistance2.length / 2] / sum2 > epsilon) {
				weightKernel[1] = DivideVectorbyLargestandLength(tempweightKernel);
				weightKernel[2] = DivideVectorbyLargestandLength(tempweightKernel2);
			} else {
				weightKernel[1] = DivideVectorbyLargestandLength(tempweightKernel);
			}
		} else if (totalTries == 3) {
			double[] tempArrayDistance1 = null;
			double[] tempArrayDistance2 = null;

			double[] tempweightKernel1 = null;
			double[] tempweightKernel2 = null;

			switch (adaptationStrategy) {
			case AdaptK:
				tempArrayDistance1 = new double[arrayDistance.length - 2];
				tempArrayDistance2 = new double[arrayDistance.length];

				for (int i = 0; i < tempArrayDistance1.length; i++) {
					tempArrayDistance1[i] = arrayDistance[i];
					tempArrayDistance2[i] = arrayDistance[i];
				}

				tempArrayDistance2[arrayDistance.length - 2] = arrayDistance[arrayDistance.length - 2];
				tempArrayDistance2[arrayDistance.length - 1] = arrayDistance[arrayDistance.length - 1];

				tempweightKernel1 = getWeightVectorKernel(tempArrayDistance1,
						kernelMethod, sigmaForKernel);

				tempweightKernel2 = getWeightVectorKernel(tempArrayDistance2,
						kernelMethod, sigmaForKernel);

				break;
			default:
				tempArrayDistance1 = arrayDistance;
				tempArrayDistance2 = arrayDistance;

				tempweightKernel1 = getWeightVectorKernel(tempArrayDistance1,
						kernelMethod, sigmaForKernel
								* (1 + differentConfigurations[0]));

				tempweightKernel2 = getWeightVectorKernel(tempArrayDistance2,
						kernelMethod, sigmaForKernel
								* (1 + differentConfigurations[2]));
			}
			weightKernel[0] = DivideVectorbyLargestandLength(tempweightKernel1);
			weightKernel[1] = DivideVectorbyLargestandLength(tempweightKernel);
			weightKernel[2] = DivideVectorbyLargestandLength(tempweightKernel2);
		} else if (totalTries == 1)
			weightKernel[0] = DivideVectorbyLargestandLength(tempweightKernel);

		return weightKernel;
	}

	/*
	 * this function returns the how the class information is distributed
	 * between instances in the neighborhood
	 */
	protected double[][] getPredictionDistributionKs(Instance instance) {
		queryInstance.inst = instance;
		lastQueryID++;
		queryInstance.queryID = lastQueryID;

		int totlalKs = kValue + 1;

		// queryComparator.queryID = lastQueryID;
		Iterator nearest = mtree.getNearestNeighbours(queryInstance);
		double[] classV = new double[totlalKs];
		double[] distanceV = new double[totlalKs];
		double totalDistances = 0;

		Instance[] neighbours = new Instance[totlalKs];
		double[] ClassC = new double[classCount];

		int countInst = 0;

		boolean exactInstanceExissts = false;
		int exactInstanceIndex = 0;
		while (nearest.hasNext() && countInst < kValue + 1) {
			InstanceInfo inf = (InstanceInfo) nearest.next();
			neighbours[countInst] = inf.inst;
			if (inf.deleted) {
				System.out.println("DELETE ERROR");
				toDelete.add(inf);
				continue;
			}

			classV[countInst] = inf.inst.classValue();
			distanceV[countInst] = distance(instance, inf.inst);

			// checking whether we found an exact instance
			if (!exactInstanceExissts && distanceV[countInst] == 0) {
				exactInstanceExissts = true;
				exactInstanceIndex = countInst;
			}
			totalDistances += distanceV[countInst];
			countInst += 1;
		}

		double[][] weightKernel = null;
		if (this.isConflictMode && EvaluationMood)
			weightKernel = geNonNormalisedWeightsFromDistances(distanceV);
		else
			weightKernel = getWeightsFromDistances(distanceV);

		double[][] result = new double[totalTries][classCount];

		int instcount2 = 0;
		for (int k = 0; k < result.length; k++) {
			if (weightKernel[k] == null) {
				result[k] = null;
				continue;
			}

			if (instance.classAttribute().isNominal()) {
				if (exactInstanceExissts
						&& exactInstanceIndex <= weightKernel.length) {
					result[k][(int) classV[exactInstanceIndex]] = 1;
					continue;
				}

				for (int i = 0; i < weightKernel[k].length; i++)
					result[k][(int) neighbours[i].classValue()] += weightKernel[k][i];// inf.inst.weight();

			} else if (instance.classAttribute().isNumeric()) {
				if (exactInstanceExissts
						&& exactInstanceIndex <= weightKernel.length) {
					result[k][0] = classV[exactInstanceIndex];
					continue;
				}

				double[] resulttemp = new double[classCount];
				double[] tempweightKernel = null;

				int numOfInstance = 0;
				try {
					for (int i = 0; i < weightKernel[k].length; i++) {
						resulttemp[0] += classV[i] * weightKernel[k][i];
					}

					numOfInstance = weightKernel[k].length;

					if (predictionStrategy == PredictionStrategy.wKNN)
						result[k][0] = resulttemp[0];
					else {
						double[][] arrayX = new double[weightKernel[k].length][];
						double[] arrayY = new double[weightKernel[k].length];

						for (int i = 0; i < weightKernel[k].length; i++) {
							arrayX[i] = NormalizeInstance(neighbours[i]);
							arrayY[i] = neighbours[i].value(classIndex);
						}

						Matrix Coefficients = Solvelinear(arrayX, arrayY,
								weightKernel[k], numOfInstance); // tempweightKernel.length

						double[] temp = NormalizeInstance(instance);
						result[k][0] = PredictLinear(Coefficients, temp);

						Range yRange = this.new Range(arrayY);
						double vRange = yRange.getBestDoubleRange(result[k][0]);

						if (EvaluationMood && DEBUG)
							System.out.println(resulttemp[0] + "\t" + result[0]
									+ "\t" + (resulttemp[0] - result[k][0]));

						// 27.12.2011
						if (EvaluationMood
								&& (Double.isNaN(result[k][0]) || Math
										.abs(vRange - result[k][0]) > EPSILON))
							result[k][0] = resulttemp[0];
					}
				}
				// in case where no solution exist for the linear System
				// e.g. when X is a singular matrix
				catch (Exception ex) {
					// ex.printStackTrace() ;
					// System.out.println("A switch is needed") ;
					if ((k == 0 && result.length == 1)
							|| (k == 1 && result.length == 3))
						result[k][0] = resulttemp[0];
					else
						result[k] = null;
				}
			} else {
				for (int i = 0; i < weightKernel[k].length; i++)
					result[k][(int) neighbours[i].classValue()] += 1;// inf.inst.weight();
			}
		}

		return result;
	}

	/*
	 * this function returns the how the class information is distributed
	 * between instances in the neighborhood
	 */
	protected double[][] getPredictionDistributionKernel(Instance instance) {
		// System.out.println("getPredictionDistributionKernel") ;
		queryInstance.inst = instance;
		lastQueryID++;
		queryInstance.queryID = lastQueryID;

		int totlalTries = differentConfigurations.length;

		// queryComparator.queryID = lastQueryID;
		Iterator nearest = mtree.getNearestNeighbours(queryInstance);
		double[] classV = new double[kValue];
		double[] distanceV = new double[kValue];
		double totalDistances = 0;

		Instance[] neighbours = new Instance[kValue];

		int countInst = 0;

		boolean exactInstanceExissts = false;
		int exactInstanceIndex = 0;

		// finding the kValue nearest neighbors
		while (nearest.hasNext() && countInst < kValue) {
			InstanceInfo inf = (InstanceInfo) nearest.next();
			neighbours[countInst] = inf.inst;
			if (inf.deleted) {
				System.out.println("DELETE ERROR");
				toDelete.add(inf);
				continue;
			}

			classV[countInst] = inf.inst.classValue();
			distanceV[countInst] = distance(instance, inf.inst);

			// checking whether we found an exact instance
			if (!exactInstanceExissts && distanceV[countInst] == 0) {
				exactInstanceExissts = true;
				exactInstanceIndex = countInst;
			}
			totalDistances += distanceV[countInst];

			countInst += 1;
		}

		double[][] weightKernel = null;
		if (this.isConflictMode && EvaluationMood)
			weightKernel = geNonNormalisedWeightsFromDistances(distanceV);
		else
			weightKernel = getWeightsFromDistances(distanceV);

		double[][] result = new double[totlalTries][classCount];

		for (int k = 0; k < result.length; k++) {
			if (weightKernel[k] == null) {
				result[k] = null;
				continue;
			}

			if (instance.classAttribute().isNominal()) {
				if (exactInstanceExissts) {
					result[k][(int) classV[exactInstanceIndex]] = 1;
					continue;
				}

				for (int i = 0; i < countInst; i++)
					result[k][(int) neighbours[i].classValue()] += weightKernel[k][i];// inf.inst.weight();

			} else if (instance.classAttribute().isNumeric()) {
				if (exactInstanceExissts) {
					result[k][0] = classV[exactInstanceIndex];
					continue;
				}

				double[] resulttemp = new double[classCount];
				int numOfInstance = 0;
				try {
					for (int i = 0; i < countInst; i++) {
						resulttemp[0] += classV[i] * weightKernel[k][i];
					}

					numOfInstance = weightKernel[k].length;

					if (predictionStrategy == PredictionStrategy.wKNN)
						result[k][0] = resulttemp[0];
					else {
						double[][] arrayX = new double[countInst][];
						double[] arrayY = new double[countInst];

						for (int i = 0; i < countInst; i++) {
							arrayX[i] = NormalizeInstance(neighbours[i]);
							arrayY[i] = neighbours[i].value(classIndex);
						}

						Matrix Coefficients = Solvelinear(arrayX, arrayY,
								weightKernel[k], numOfInstance); // tempweightKernel.length

						double[] temp = NormalizeInstance(instance);
						result[k][0] = PredictLinear(Coefficients, temp);

						Range yRange = this.new Range(arrayY);
						double vRange = yRange.getBestDoubleRange(result[k][0]);

						if (EvaluationMood && DEBUG)
							System.out.println(resulttemp[0] + "\t" + result[0]
									+ "\t" + (resulttemp[0] - result[k][0]));

						// 27.12.2011
						if (EvaluationMood
								&& (Double.isNaN(result[k][0]) || (Math
										.abs(vRange - result[k][0])) > EPSILON))
							result[k][0] = resulttemp[0];

					}
				}
				// in case where no solution exist for the linear System
				// e.g. when X is a singular matrix
				catch (Exception ex) {
					// ex.printStackTrace() ;
					// System.out.println("A switch is needed") ;
					if ((k == 0 && result.length == 1)
							|| (k == 1 && result.length == 3))
						result[k][0] = resulttemp[0];
					else
						result[k] = null;
				}
			} else {
				for (int i = 0; i < countInst; i++)
					result[k][(int) neighbours[i].classValue()] += 1;// inf.inst.weight();
			}
		}
		return result;
	}

	/**
	 * Compute distance of two Instances
	 * 
	 * @param instance1
	 *            first instance
	 * @param instance2
	 *            second instance
	 * @return distance of instances
	 */
	protected double distance(Instance instance1, Instance instance2) {
		double dist = 0;
		for (int i = 0; i < allAttributesCount; i++) {
			if (usedAttribute[i] > 0) {
				double val1 = 0, val2 = 0;
				if (instance1.isMissing(i))
					val1 = Double.NaN;
				else
					val1 = instance1.value(i);
				if (instance2.isMissing(i))
					val2 = Double.NaN;
				else
					val2 = instance2.value(i);
				// double d = distance.distance(i, val1, val2);
				// if (d < 0 || d > 1) System.out.println("ERROR!!!");
				// dist += d * d;
				// It returns d * d
				dist += distance.sqDifference(i, val1, val2);
			}
		}
		return Math.sqrt(dist);
	}

	/**
	 * Detect the attributes to use for learning.
	 */
	protected void computeUsedAttributes() {
		usedAttribute = new int[allAttributesCount];
		for (int i = 0; i < allAttributesCount; i++) {
			if (i != classIndex && i != timeIndex) {
				Attribute att = header.attribute(i);
				if (att.isNominal()) {
					usedAttribute[i] = NOMINAL;
				} else if (att.isNumeric())
					usedAttribute[i] = NUMERIC;
			}
		}
	}

	@Override
	public Measurement[] getModelMeasurementsImpl() {

		return new Measurement[] {
				new Measurement("IBLStreams size", currentSize),
				new Measurement("sigmaForKernel", sigmaForKernel),
				new Measurement("kValue", kValue),
				new Measurement("NumOfDrifts", NumOfDrifts)

		};
	}

	// this function is used to add a time index as the last attribute for the
	// whole data set
	// private Instances AddTimeIndex(Instances data) {
	// Attribute timeAtt = new Attribute("time");
	// data.insertAttributeAt(timeAtt, data.numAttributes());
	//
	// timeIndex = data.numAttributes() - 1;
	// for (int i = 0; i < data.numInstances(); i++) {
	// data.get(i).setValue(timeIndex, timeSequence++);
	// }
	// return data;
	// }

	/**
	 * This function is used to add a time index as the last attribute for the
	 * whole data set. Modified by Álvar because insertAttributeAt is not yet
	 * implemented.
	 * 
	 * @param data
	 *            Instances to add the new attribute.
	 * @return Instances with the new attribute added.
	 */
	private Instances AddTimeIndex(Instances data) {
		List<Attribute> attributes = new ArrayList<>();
		Instances newData;
		Instance inst;

		timeIndex = data.numAttributes() - 1;

		for (int i = 0; i < data.numAttributes(); i++) {
			if (i == timeIndex)
				attributes.add(new Attribute("time"));

			attributes.add(data.attribute(i));
		}

		newData = new Instances(data.getRelationName(), attributes,
				data.numInstances());
		newData.setClassIndex(data.classIndex() + 1);
		newData.delete();

		for (int i = 0; i < data.numInstances(); i++) {
			inst = new DenseInstance(newData.numAttributes());

			for (int j = 0, k = 0; j < data.numAttributes(); j++, k++) {
				if (j == timeIndex) {
					inst.setValue(timeIndex, timeSequence++);
					j++;
				}

				inst.setValue(j, data.get(i).value(k));
			}

			inst.setDataset(newData);
			newData.add(inst);
		}
		data = newData;
		return newData;
	}

	// this function is used to add a time index as the last attribute of an
	// instance
	// private Instance AddTimeIndex(Instance ins) {
	// Instances instances = ins.dataset();
	// ins.setDataset(null);
	// ins.insertAttributeAt(ins.numAttributes());
	// ins.setDataset(instances);
	//
	// ins.setValue(timeIndex, timeSequence++);
	//
	// return ins;
	// }

	/**
	 * This function is used to add a time index as the last attribute of an
	 * instance. Modified by Álvar because insertAttributeAt is not yet
	 * implemented.
	 * 
	 * @param ins
	 *            Instance to add the new attribute.
	 * @return Instance with the new attribute added.
	 */
	private Instance AddTimeIndex(Instance ins, boolean addNewAtt) {
		List<Attribute> attributes = new ArrayList<>();
		Instances newData;
		Instance newInst;

		if (addNewAtt) {
			for (int i = 0; i < ins.numAttributes(); i++) {
				if (i == timeIndex)
					attributes.add(new Attribute("time"));

				attributes.add(ins.attribute(i));
			}

			newData = new Instances(ins.dataset().getRelationName(),
					attributes, ins.dataset().numInstances());
			newData.setClassIndex(ins.dataset().classIndex());
			newData.setClassIndex(ins.classIndex() + 1);
			newData.delete();

			newInst = new DenseInstance(attributes.size());
			newInst.setDataset(newData);
		} else {
			newData = ins.dataset();
			newInst = new DenseInstance(ins.dataset().numAttributes());
			newInst.setDataset(newData);
		}

		for (int j = 0, k = 0; j < ins.numAttributes(); j++, k++) {
			if (j == timeIndex) {
				newInst.setValue(timeIndex, timeSequence++);
				j++;
			}

			newInst.setValue(j, ins.value(k));
		}

		return newInst;
	}

	public static void main(String[] str) {
		IBLStreams iib = new IBLStreams();
		double[] dob = { 4, -12.2, 1.5, 23.3, -1, 25 };
		Range rng = iib.new Range(dob);

		System.out.println(rng.getBestDoubleRange(-100));
		System.out.println(rng.getBestDoubleRange(-50));
		System.out.println(rng.getBestDoubleRange(-25));
		System.out.println(rng.getBestDoubleRange(-20));
		System.out.println(rng.getBestDoubleRange(-10));
		System.out.println(rng.getBestDoubleRange(-5));
		System.out.println(rng.getBestDoubleRange(0));
		System.out.println(rng.getBestDoubleRange(5));
		System.out.println(rng.getBestDoubleRange(10));
		System.out.println(rng.getBestDoubleRange(15));
		System.out.println(rng.getBestDoubleRange(20));
		System.out.println(rng.getBestDoubleRange(25));
		System.out.println(rng.getBestDoubleRange(40));
		System.out.println(rng.getBestDoubleRange(50));
		System.out.println(rng.getBestDoubleRange(150));

		System.exit(0);

		double ZalphaDiv2 = 1.645;

		for (int k = 0; k < 100; k++) {
			Random generator = new Random(k);

			double[] array = new double[10];

			for (int i = 0; i < array.length; i++)
				array[i] = generator.nextDouble();

			IBLStreams ib = new IBLStreams();
			IBLStreams.ConfidenceInterval conf = ib.new ConfidenceInterval(
					array, ZalphaDiv2);

			boolean allok = true;
			for (int i = 0; i < array.length; i++)
				if (!conf.belongToInterval(array[i]))
					allok = false;

			System.out.println(allok);
		}

		double[] array = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6 };
		IBLStreams ib = new IBLStreams();
		IBLStreams.ConfidenceInterval conf = ib.new ConfidenceInterval(array,
				ZalphaDiv2);

		boolean allok = true;
		for (int i = 0; i < array.length; i++)
			if (!conf.belongToInterval(array[i]))
				allok = false;

		System.out.println(allok);
	}

	/*
	 * this class hold the information of a confidence interval generated a
	 * single array of double values
	 */
	class ConfidenceInterval {
		public final double mean;
		public final double variance;
		public final double sdv;
		public final double lowerBound;
		public final double upperBound;

		public ConfidenceInterval(double[] values, double ZalphaDiv2) {
			// TODO Auto-generated constructor stub

			double temp = 0;
			for (int i = 0; i < values.length; i++) {
				temp += values[i];
			}
			mean = temp / values.length;

			temp = 0;
			for (int i = 0; i < values.length; i++) {
				temp += Math.pow(values[i] - mean, 2);
			}
			variance = temp / values.length;
			sdv = Math.sqrt(variance);

			// 29.12.2011
			// lowerBound= mean - (ZalphaDiv2 * sdv) ;
			// upperBound= mean + (ZalphaDiv2 * sdv) ;

			lowerBound = mean - (ZalphaDiv2 * sdv / Math.sqrt(values.length));
			upperBound = mean + (ZalphaDiv2 * sdv / Math.sqrt(values.length));
		}

		public boolean belongToInterval(double value) {

			if (value >= lowerBound && value <= upperBound)
				return true;
			else
				return false;
		}

		public String toString() {
			String str = "";
			str += "Lo=" + lowerBound + "\tUp=" + upperBound;

			return str;
		}
	}

	// this function calculates the weights of a set of instances based on their
	// d
	// distances to a given instance
	// inarray: the distances of the neighbor instances
	// WeightingSchemeKernel : is the wished weighting scheme
	// sigma is the variance used for both Gaussian and exponential kernels
	// not the returned weights here are not normalized

	public static double[] getWeightVectorKernel(double[] inarray,
			WeightingSchemeKernel scheme, double sigma) {
		double[] result = new double[inarray.length];
		double[] tempKern = new double[inarray.length];
		double totalKern = 0;

		double SigmaSq2 = 2 * Math.pow(sigma, 2);

		switch (scheme) {
		case GaussianKernel: {
			for (int i = 0; i < inarray.length; i++) {
				tempKern[i] = Math.exp(-Math.pow(inarray[i], 2) / SigmaSq2);
				totalKern += tempKern[i];
			}

			break;
		}

		case exponentialKernel: {
			for (int i = 0; i < inarray.length; i++) {
				tempKern[i] = Math.exp(-inarray[i] / SigmaSq2);
				totalKern += tempKern[i];
			}
			break;
		}

		case equal: {
			for (int i = 0; i < result.length; i++) {
				tempKern[i] = (double) 1 / inarray.length;
			}
			break;
		}
		case inverseDistance: {
			for (int i = 0; i < inarray.length; i++) {
				tempKern[i] = 1 / (inarray[i]);
			}
			break;
		}
		case linear: {
			double wk = 0.001;
			double dk = Double.MIN_VALUE;

			for (int i = 0; i < inarray.length; i++) {
				if (dk < inarray[i])
					dk = inarray[i];
			}
			for (int i = 0; i < inarray.length; i++) {
				tempKern[i] = wk + (1 - wk) * (dk - inarray[i]) / dk;
			}
			break;
		}
		}

		return tempKern;
	}

	// this function is used to normalize an instance,
	// depending on the statistics extracted from the already seen instances

	public double[] NormalizeInstance(Instance inst) {
		int attCount = 0;
		for (int i = 0; i < usedAttribute.length; i++) {
			if (usedAttribute[i] == NOMINAL || usedAttribute[i] == NUMERIC)
				attCount++;
		}

		double result[] = new double[attCount];
		attCount = 0;
		for (int i = 0; i < usedAttribute.length; i++) {
			if (usedAttribute[i] == NUMERIC) {
				result[attCount] = (inst.value(i) - AttMinVal[i])
						/ (AttMaxVal[i] - AttMinVal[i]);
				attCount++;
			}
		}
		return result;
	}

	// this function is used to normalize a vector of double Values
	public double[] NormalizeVector(double[] vals) {
		double totalKern = 0;

		for (int i = 0; i < vals.length; i++) {
			totalKern += vals[i];
		}

		for (int i = 0; i < vals.length; i++) {
			vals[i] = vals[i] / totalKern;
		}

		return vals;
	}

	public double[] DivideVectorbyLargest(double[] vals) {
		double max = -Double.MAX_VALUE;

		for (int i = 0; i < vals.length; i++) {
			if (max < vals[i])
				max = vals[i];
		}

		for (int i = 0; i < vals.length; i++) {
			vals[i] = vals[i] / max;
		}

		return vals;
	}

	public double[] DivideVectorbyLargestandLength(double[] vals) {
		double max = -Double.MAX_VALUE;

		for (int i = 0; i < vals.length; i++) {
			if (max < vals[i])
				max = vals[i];
		}

		for (int i = 0; i < vals.length; i++) {
			vals[i] = vals[i] / (max * vals.length);
		}

		return vals;
	}

	// this function is used to return the average of a double Values
	public double avergeVector(double[] vals) {
		double totalKern = 0;

		for (int i = 0; i < vals.length; i++) {
			totalKern += vals[i];
		}

		return totalKern / vals.length;
	}

	// this function is used to extract the statistics from the already seen
	// instances
	protected void addInstanceStatisitcs(Instance inst) {

		for (int attr = 0; attr < usedAttribute.length; attr++) {

			if (usedAttribute[attr] != NOMINAL
					&& usedAttribute[attr] != NUMERIC)
				continue;

			if (inst.isMissing(attr))
				continue;

			if (usedAttribute[attr] == NUMERIC) {
				if (AttMinVal[attr] > inst.value(attr)) {
					AttMinVal[attr] = inst.value(attr);
				}
				if (AttMaxVal[attr] < inst.value(attr)) {
					AttMaxVal[attr] = inst.value(attr);
				}
			} else if (usedAttribute[attr] == NOMINAL) {
				distribs[attr][(int) inst.value(attr)][(int) inst.classValue()] += 1;
				if (Double
						.isNaN(distribs[attr][(int) inst.value(attr)][(int) inst
								.classValue()]))
					System.out.println("NAN !!!!");
			}
		}
	}

	// this function is used to solve a linear regression
	public Matrix Solvelinear(double[][] tempArrayX, double[] tempArrayY,
			double[] tempArraywight, int numofInst) {
		int numOfInst = numofInst; // tempArrayX.length ;
		int numOfDim = tempArrayX[0].length;

		double[][] arrayX = new double[numOfInst][numOfDim + 1];
		double[][] arrayY = new double[numOfInst][1];
		double[][] arrayW = new double[numOfInst][numOfInst];

		for (int i = 0; i < numOfInst; i++) {
			for (int j = 0; j < numOfDim; j++) {
				arrayX[i][j] = tempArrayX[i][j];
			}
			arrayX[i][numOfDim] = 1;
			arrayY[i][0] = tempArrayY[i];
			arrayW[i][i] = tempArraywight[i];
		}

		Matrix X = new Matrix(arrayX);
		Matrix Y = new Matrix(arrayY);
		Matrix W = new Matrix(arrayW);

		if (EvaluationMood && DEBUG) {
			System.out.println("X\t=" + printArray(X));
			System.out.println("X'\t=" + printArray(X.transpose()));
			System.out.println("Y\t=" + printArray(Y));
			System.out.println("Y'\t=" + printArray(Y.transpose()));
			System.out.println("W\t=" + printArray(W));
		}

		Matrix temp = X.transpose().times(W).times(X);

		if (EvaluationMood && DEBUG) {
			System.out.println("temp\t=" + printArray(temp));
			System.out.println("temp.inverse\t=" + printArray(temp.inverse()));
		}

		temp = temp.inverse().times(X.transpose().times(W).times(Y));

		if (EvaluationMood && DEBUG)
			System.out.println("temp\t=" + printArray(temp));

		return temp;
	}

	// this function uses the solution of the linear regression
	// problem to predict the target value for a new instance
	public static double PredictLinear(Matrix Coefficients, double[] instance) {
		double result = 0;

		for (int i = 0; i < instance.length; i++) {
			result += Coefficients.get(i, 0) * instance[i];
		}

		result += Coefficients.get(Coefficients.getRowDimension() - 1, 0);
		return result;
	}

	/*
	 * // this function is used to solve a polynomial regression 2nd degree
	 * public static Matrix SolvePolynomial2Degree(double[][] tempArrayX
	 * ,double[] tempArrayY ) { int numOfDim= tempArrayX[0].length ; int
	 * numOfCof= numOfDim + (numOfDim +1) * (numOfDim) / 2 +1 ;
	 * 
	 * double[][] arrayX = new double [tempArrayX.length][numOfCof] ; double[][]
	 * arrayY = new double [tempArrayX.length][1] ;
	 * 
	 * for (int i = 0 ; i < tempArrayX.length ; i++) { int coIndex= 0 ; for (int
	 * j = 0 ; j < numOfDim; j++) { arrayX[i][coIndex] = tempArrayX[i][j] ;
	 * coIndex++ ; for (int k = j ; k < numOfDim; k++) { arrayX[i][coIndex] =
	 * tempArrayX[i][j] * tempArrayX[i][k]; coIndex++ ; } } arrayX[i][coIndex] =
	 * 1 ; arrayY[i][0] = tempArrayY[i] ; }
	 * 
	 * Matrix X = new Matrix(arrayX); Matrix Y = new Matrix(arrayY) ;
	 * 
	 * Matrix temp= X.transpose().times(X) ;
	 * temp=temp.inverse().times(X.transpose()) ; temp=temp.times(Y) ;
	 * 
	 * return temp ; }
	 * 
	 * 
	 * // this function uses the solution of the polynomial regression 2nd
	 * degree // problem to predict the target value for a new instance public
	 * static double PredictPolynomial2Degree(Matrix Coefficients, double []
	 * instance) { int numOfDim= instance.length ; int numOfCof=
	 * Coefficients.getRowDimension() ;
	 * 
	 * double result=0 ;
	 * 
	 * int coIndex= 0 ; for (int j = 0 ; j < numOfDim; j++) { result +=
	 * instance[j] * Coefficients.get(coIndex, 0) ; coIndex++ ; for (int k = j ;
	 * k < numOfDim; k++) { result += instance[j] * instance[k] *
	 * Coefficients.get(coIndex, 0) ; coIndex++ ; } } result +=
	 * Coefficients.get(coIndex, 0) ; return result ; }
	 */

	public enum WeightingSchemeKernel {
		equal, inverseDistance, linear,

		GaussianKernel, exponentialKernel
	}

	public enum AdaptationStrategy {
		AdaptK, AdaptSigma, none
	}

	public enum PredictionStrategy {
		wKNN, LinearRegression
	}

	// this function is used to print a matrix
	public static String printArray(Matrix array) {

		int numOfR = array.getRowDimension();
		int numOfC = array.getColumnDimension();

		String result = "";

		int coIndex = 0;
		for (int i = 0; i < numOfR; i++) {
			for (int j = 0; j < numOfC; j++) {
				result += array.get(i, j) + "\t";
			}
			result += "\n";
		}
		return result;
	}

	// this function is used to sum up the elements of a one dimensional double
	// array
	public static double sumArrayElements(double[] vals) {
		double result = 0;

		for (int i = 0; i < vals.length; i++) {
			result += vals[i];
		}
		return result;
	}

	// this function is used to get the index of the
	// maximum value in an array
	public static int getIndexofMaxElement(double[] vals) {
		double maxValue = Double.MIN_VALUE;
		int maxIndex = 0;

		for (int i = 0; i < vals.length; i++) {
			if (vals[i] > maxValue) {
				maxIndex = i;
				maxValue = vals[i];
			}
		}
		return maxIndex;
	}

	// this function is used to get the index of the
	// median value in an array
	public static int getIndexofMedianElement(double[] vals) {
		double sum = 0;
		for (int i = 0; i < vals.length; i++) {
			sum += vals[i];
		}

		sum /= 2;

		double sumTemp = 0;
		int medianIndex = 0;

		for (int i = 0; i < vals.length; i++) {
			sumTemp += vals[i];
			if (sumTemp >= sum) {
				medianIndex = i;
				break;
			}
		}
		return medianIndex;
	}

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub

	}

	class Range {
		private double min = Double.POSITIVE_INFINITY;
		private double max = Double.NEGATIVE_INFINITY;

		public Range(double[] array) {
			for (int i = 0; i < array.length; i++) {
				if (array[i] > max)
					max = array[i];
				else if (array[i] < min)
					min = array[i];
			}
		}

		public double getBestDoubleRange(double value) {
			double result = 0;
			double rangeLength = max - min;
			if (value > (max + rangeLength / 2))
				result = max;
			else if (value < (min - rangeLength / 2))
				result = min;
			else
				result = value;

			return result;
		}
	}
}