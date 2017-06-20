package weka.core;
/*
 * Created on 20.08.2004
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Generation - Code and Comments
 */


import java.io.Serializable;
import java.util.Enumeration;
import java.util.Iterator;

import utils.stats.Distribution;

/**
 * Class of a distance measure. The distance of a numeric attribute can be
 * computed with no normalization, with range normalization (maximum difference
 * is normalized to 1 (max-min)) and with a quantile normalization. The distance
 * of a nominal attribute is computed with the Value Difference Metric (VDM)
 * which was introduced by Stanfill and Waltz (1986) or with the simple 0-1
 * Distance (if equal 0, else 1).
 */
public class NormDistance implements Cloneable, Serializable{

	private static final long serialVersionUID = -4852995595042374918L;

	/** Constant for a nominal Attribute */
    public static int NOMINAL = 2;

    /** Constant for a numeric Attribute */
    public static int NUMERIC = 1;

    /** Constant for no numeric normalization */
    public static int NO_NUMERIC = 0;

    /** Constant for range normalization of numeric attributes */
    public static int RANGE_NUMERIC = 1;

    /** Constant for quantile range normalization of numeric attributes */
    public static int QUANTILE_NUMERIC = 2;
    
    
    public static double maxDistance(NormDistance norm1, NormDistance norm2){
        if (norm1.useVDM!=norm2.useVDM || norm1.numeric!=norm2.numeric || !norm1.instances.equalHeaders(norm2.instances)) return -1;
        double max=0;
        for (int i = 0; i < norm1.usedAttributes.length; i++) {
            int attr = norm1.usedAttributes[i];
            if (norm1.isUsedAttributes[attr] == NOMINAL && norm1.useVDM) {
                int valCount = norm1.instances.attribute(attr).numValues();
                
                for (int val = 1; val < valCount; val++) {
                    for (int val2 = 0; val2 < val; val2++) {
                        max=Math.max(max,Math.abs(norm1.distNominal[attr][val][val2]-norm2.distNominal[attr][val][val2]));
                    }
                }
            } else if (norm1.isUsedAttributes[attr] == NUMERIC
                    && (norm1.numeric == QUANTILE_NUMERIC ||  norm1.numeric == RANGE_NUMERIC)) {
                max=Math.max(max,Math.abs(1.0-norm1.maxDiff[attr]/norm2.maxDiff[attr]));
                max=Math.max(max,Math.abs(1.0-norm2.maxDiff[attr]/norm1.maxDiff[attr]));
            } 
        }
        return max;
    }
    
    

    /** Header of data */
    protected Instances instances;

    /** List of used attributes for distance computation */
    protected int[] usedAttributes;

    /** Array of used attributes */
    protected int[] isUsedAttributes;

    /** minimum value of the numeric attributes */
    protected double[] min;

    /** maximum value of the numeric attributes */
    protected double[] max;

    /** maximum difference of the numeric attributes */
    protected double[] maxDiff;

    /** Distributions of the numeric attributes to compute the quantile value. */
    protected Distribution[] distributions;

    /** Quantile-value for quantile range normalization. */
    protected double quantile;

    /** class distribution of the nominal attributes */
    public double[][][] distribs;

    /** distance matrices of the nominal attributes */
    protected double[][][] distNominal;

    /** if numeric variables are normalized to max difference of 1 */
    protected int numeric = 0;

    /** if vdm is used for nominal attributes */
    protected boolean useVDM;

    /**
     * Create Normdistance for the instances and the attributes.
     * 
     * @param instances
     *            Instances
     * @param usedAttributes
     *            used attributes
     */
    public NormDistance(Instances instances, int[] usedAttributes) {
        this(instances, usedAttributes, true, true);
    }

    /**
     * Create Normdistance for the instances and the attributes.
     * 
     * @param instances
     *            Instances
     * @param usedAttributes
     *            used attributes with kind of attribute (numeric/nominal)
     * @param normalize
     *            whether numeric attributes should normalized
     * @param useVDM
     *            whether vdm measure should be used for nominal attributes
     */
    public NormDistance(Instances instances, int[] usedAttributes,
            boolean rangeNumeric, boolean useVDM) {
        this.instances = instances;
        this.numeric = rangeNumeric ? RANGE_NUMERIC : NO_NUMERIC;
        this.useVDM = useVDM;
        FastVector vec = new FastVector();
        for (int i = 0; i < usedAttributes.length; i++) {
            if (usedAttributes[i] > 0)
                vec.addElement(new Integer(i));
        }
        this.usedAttributes = new int[vec.size()];
        for (int i = 0; i < vec.size(); i++) {
            this.usedAttributes[i] = ((Integer) vec.elementAt(i)).intValue();
        }
        this.isUsedAttributes = usedAttributes;
        norm(instances);
    }

    /**
     * Create Normdistance for the instances and the attributes.
     * 
     * @param instances
     *            Instances
     * @param usedAttributes
     *            used attributes
     * @param normalize
     *            whether numeric attributes should normalized
     * @param useVDM
     *            whether vdm measure should be used for nominal attributes
     */
    public NormDistance(Instances instances, boolean[] usedAttributes,
            boolean rangeNumeric, boolean useVDM) {
        this.instances = instances;
        this.numeric = rangeNumeric ? RANGE_NUMERIC : NO_NUMERIC;
        this.useVDM = useVDM;
        FastVector vec = new FastVector();
        for (int i = 0; i < usedAttributes.length; i++) {
            if (usedAttributes[i])
                vec.addElement(new Integer(i));
        }
        this.usedAttributes = new int[vec.size()];
        for (int i = 0; i < vec.size(); i++) {
            this.usedAttributes[i] = ((Integer) vec.elementAt(i)).intValue();
        }
        this.isUsedAttributes = new int[usedAttributes.length];
        for (int i = 0; i < usedAttributes.length; i++) {
            if (instances.attribute(i).isNominal())
                isUsedAttributes[i] = NOMINAL;
            else if (instances.attribute(i).isNumeric())
                isUsedAttributes[i] = NUMERIC;
        }
        norm(instances);
    }

    /**
     * Create Normdistance for the instances and the attributes.
     * 
     * @param instances
     *            Instances
     * @param usedAttributes
     *            used attributes with kind of attribute (numeric/nominal)
     * @param normalize
     *            whether numeric attributes should normalized
     * @param useVDM
     *            whether vdm measure should be used for nominal attributes
     */
    public NormDistance(Instances instances, int[] usedAttributes,
            double quantile, boolean useVDM) {
        this.instances = instances;
        this.numeric = QUANTILE_NUMERIC;
        this.quantile = quantile;
        this.useVDM = useVDM;
        FastVector vec = new FastVector();
        for (int i = 0; i < usedAttributes.length; i++) {
            if (usedAttributes[i] > 0)
                vec.addElement(new Integer(i));
        }
        this.usedAttributes = new int[vec.size()];
        for (int i = 0; i < vec.size(); i++) {
            this.usedAttributes[i] = ((Integer) vec.elementAt(i)).intValue();
        }
        this.isUsedAttributes = usedAttributes;
        norm(instances);
    }

    /**
     * Create Normdistance for the instances and the attributes.
     * 
     * @param instances
     *            Instances
     * @param usedAttributes
     *            used attributes
     * @param normalize
     *            whether numeric attributes should normalized
     * @param useVDM
     *            whether vdm measure should be used for nominal attributes
     */
    public NormDistance(Instances instances, boolean[] usedAttributes,
            double quantile, boolean useVDM) {
        this.instances = instances;
        this.numeric = QUANTILE_NUMERIC;
        this.quantile = quantile;
        this.useVDM = useVDM;
        FastVector vec = new FastVector();
        for (int i = 0; i < usedAttributes.length; i++) {
            if (usedAttributes[i])
                vec.addElement(new Integer(i));
        }
        this.usedAttributes = new int[vec.size()];
        for (int i = 0; i < vec.size(); i++) {
            this.usedAttributes[i] = ((Integer) vec.elementAt(i)).intValue();
        }
        this.isUsedAttributes = new int[usedAttributes.length];
        for (int i = 0; i < usedAttributes.length; i++) {
            if (instances.attribute(i).isNominal())
                isUsedAttributes[i] = NOMINAL;
            else if (instances.attribute(i).isNumeric())
                isUsedAttributes[i] = NUMERIC;
        }
        norm(instances);
    }

    

    /**
     * Compute normalization for this instances.
     * 
     * @param instances
     *            data to normalize
     */
    public void norm(Instances instances) {
        if (!useVDM && numeric == NO_NUMERIC)
            return;
        initalize();
        // evaluate instances
        Enumeration enum1 = instances.enumerateInstances();
        while (enum1.hasMoreElements()) {
            Instance inst = (Instance) enum1.nextElement();
            add(inst);
        }
        doFlush();
    }

    /**
     * Compute normalization for this instances.
     * 
     * @param instances
     *            data to normalize
     */
    public void norm(Iterator<Instance> instances) {
        if (!useVDM && numeric == NO_NUMERIC)
            return;
        initalize();
        // evaluate instances
        while (instances.hasNext()) {
            Instance inst = (Instance) instances.next();
            add(inst);
        }
        doFlush();
    }

    /**
     * Add an instance into the distribution
     * 
     * @param inst
     */
    public void addInstance(Instance inst) {
        add(inst);
        doFlush();
    }

    /**
     * Add instances into the distribution
     * 
     * @param inst
     */
    public void addInstances(Iterator<Instance> it) {
        while (it.hasNext())
            add(it.next());
        doFlush();
    }

    /**
     * remove an instance of the data
     * 
     * @param inst
     *            the instance
     */
    public void removeInstance(Instance inst) {
        remove(inst);
        doFlush();
    }

    /**
     * remove instances of the data
     * 
     * @param inst
     *            the instance
     */
    public void removeInstances(Iterator<Instance> it) {
        while (it.hasNext())
            remove(it.next());
        doFlush();
    }

    /**
     * Add a simple instance.
     * 
     * @param inst
     *            the instance
     */
    protected void add(Instance inst) {
        if (!useVDM && numeric == NO_NUMERIC)
            return;
        for (int i = 0; i < usedAttributes.length; i++) {
            int attr = usedAttributes[i];
            if (inst.isMissing(attr))
                continue;
            if (isUsedAttributes[attr] == NUMERIC) {
                if (numeric == RANGE_NUMERIC) {
                    if (min[attr] > inst.value(attr)) {
                        min[attr] = inst.value(attr);
                    }
                    if (max[attr] < inst.value(attr)) {
                        max[attr] = inst.value(attr);
                    }
                } else if (numeric == QUANTILE_NUMERIC) {
                    distributions[attr].add(inst.value(attr));
                }
            } else if (isUsedAttributes[attr] == NOMINAL && useVDM
                    ) {
                distribs[attr][(int) inst.value(attr)][(int) inst.classValue()] += 1;
                if (Double
                        .isNaN(distribs[attr][(int) inst.value(attr)][(int) inst
                                .classValue()]))
                    System.out.println("NAN !!!!");
            }
        }
    }

    /**
     * Remove an instance.
     * 
     * @param inst
     *            the instance
     */
    protected void remove(Instance inst) {
        if (!useVDM && numeric == NO_NUMERIC)
            return;
        for (int i = 0; i < usedAttributes.length; i++) {
            int attr = usedAttributes[i];
            if (inst.isMissing(attr))
                continue;
            if (isUsedAttributes[attr]==NOMINAL && useVDM) {
                if (Double
                        .isNaN(distribs[attr][(int) inst.value(attr)][(int) inst
                                .classValue()]))
                    System.out.println("NAN !!!!");
                distribs[attr][(int) inst.value(attr)][(int) inst.classValue()] -= 1;
                if (distribs[attr][(int) inst.value(attr)][(int) inst
                        .classValue()] < 0)
                    System.out.println("<0!!!");
                if (Double
                        .isNaN(distribs[attr][(int) inst.value(attr)][(int) inst
                                .classValue()]))
                    System.out.println("NAN !!!!");
            }
        }
    }
    
    public Object clone(){
        NormDistance cl=null;
        try {
            cl = (NormDistance)super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        cl.initalize();
        for (int i = 0; i < usedAttributes.length; i++) {
            int attr = usedAttributes[i];
            if (isUsedAttributes[attr] == NOMINAL && useVDM) {
                int valCount = instances.attribute(attr).numValues();
                
                for (int val = 0; val < valCount; val++) {
                    
                    for (int cla = 0; cla < instances.numClasses(); cla++) {
                        cl.distribs[attr][val][cla]=distribs[attr][val][cla];
                    }
                    for (int val2 = 0; val2 < val; val2++) {
                        cl.distNominal[attr][val][val2]=distNominal[attr][val][val2];
                        cl.distNominal[attr][val2][val]=distNominal[attr][val2][val];
                    }
                }
            } else if (isUsedAttributes[attr] == NUMERIC
                    && (numeric == QUANTILE_NUMERIC ||  numeric == RANGE_NUMERIC)) {
                cl.maxDiff[attr]=maxDiff[attr];
                cl.min[attr]=min[attr];
                cl.max[attr]=max[attr];
            } 
        }
        return cl;
    }

    /**
     * Initialize the variables to collect data.
     */
    protected void initalize() {
        if (!useVDM && numeric == NO_NUMERIC)
            return;
        int attrCount = instances.numAttributes();
        min = new double[attrCount];
        max = new double[attrCount];
        maxDiff = new double[attrCount];
        distribs = new double[attrCount][][];
        distNominal = new double[attrCount][][];
        distributions = new Distribution[attrCount];
        // initialize fields
        for (int i = 0; i < attrCount; i++) {
            if (isUsedAttributes[i] == NUMERIC) {
                if (numeric == RANGE_NUMERIC) {
                    min[i] = Double.MAX_VALUE;
                    max[i] = -Double.MAX_VALUE;
                } else if (numeric == QUANTILE_NUMERIC) {
                    distributions[i] = new Distribution();
                }
            } else if (isUsedAttributes[i] == NOMINAL && useVDM) {
                distribs[i] = new double[instances.attribute(i).numValues()][instances
                        .numClasses()];
                distNominal[i] = new double[instances.attribute(i).numValues() + 1][instances
                        .attribute(i).numValues() + 1];
            }
        }
    }

    /**
     * Flush data to compute the distances of data.
     */
    protected void doFlush() {
        if (!useVDM && numeric == NO_NUMERIC)
            return;
        // compute distances and values
        for (int i = 0; i < usedAttributes.length; i++) {
            int attr = usedAttributes[i];
            if (isUsedAttributes[attr] == NOMINAL && useVDM) {
                int valCount = instances.attribute(attr).numValues();
                double total = 0;
                double[] distrib = new double[instances.numClasses()];
                double[][] distribVal = new double[valCount][instances
                        .numClasses()];
                for (int val = 0; val < valCount; val++) {
                    double sum = 0;

                    for (int cl = 0; cl < instances.numClasses(); cl++) {
                        sum += distribs[attr][val][cl];
                        distribVal[val][cl] = distribs[attr][val][cl];
                        distrib[cl] += distribs[attr][val][cl];
                    }
                    total += sum;
                    if (sum > 0) {
                        for (int cl = 0; cl < instances.numClasses(); cl++) {
                            distribVal[val][cl] /= sum;
                        }
                    } else {
                        distribVal[val] = distrib;
                        
                    }
                    distNominal[attr][val][val] = 0;
                    
                }
                for (int cl = 0; cl < instances.numClasses(); cl++) {
                    distrib[cl] /= total;
                }
                for (int val = 0; val < valCount; val++) {
                    for (int val2 = 0; val2 < val; val2++) {
                        distNominal[attr][val][val2] = distribDistance(
                                distribVal[val], distribVal[val2]);
                        distNominal[attr][val2][val] = distNominal[attr][val][val2];
                    }
                }
                double sum = 0;
                for (int val = 0; val < valCount; val++) {
                    double dist = distribDistance(distribVal[val], distrib);
                    distNominal[attr][val][valCount] = dist;
                    distNominal[attr][valCount][val] = dist;
                    sum += dist;
                }
                distNominal[attr][valCount][valCount] = sum / valCount;
            } else if (isUsedAttributes[attr] == NUMERIC
                    && numeric == QUANTILE_NUMERIC) {
                min[attr] = distributions[attr].getQuantile(quantile);
                max[attr] = distributions[attr].getQuantile(1.0 - quantile);
                maxDiff[attr] = max[attr] - min[attr];
                if (maxDiff[attr] <= 0.0) {
                    min[attr] = distributions[attr].getQuantile(0.0);
                    max[attr] = distributions[attr].getQuantile(1.0);
                    maxDiff[attr] = max[attr] - min[attr];
                    if (maxDiff[attr] <= 0.0)
                        maxDiff[attr] = 1.0;
                }
            } else if (isUsedAttributes[attr] == NUMERIC
                    && numeric == RANGE_NUMERIC) {
                maxDiff[attr] = max[attr] - min[attr];
                if (maxDiff[attr] <= 0.0)
                    maxDiff[attr] = 1.0;
            }
        }
    }

    /**
     * Delete variables if no further instances will be added.
     */
    public void doFinalize() {
        if (useVDM)
            distribs = null;
        if (numeric == QUANTILE_NUMERIC)
            distributions = null;
    }

    /**
     * Compute distance of the simple arrays
     * 
     * @param dist1
     *            first array
     * @param dist2
     *            second array
     * @return distance
     */
    protected double distribDistance(double[] dist1, double[] dist2) {
        double dist = 0;
        for (int i = 0; i < dist1.length; i++) {
            dist += Math.abs(dist1[i] - dist2[i]);
        }
        return dist;
    }

    public double getMinimum(int attribute) {
        return min[attribute];
    }

    public double getMaximum(int attribute) {
        return max[attribute];
    }

    /**
     * Compute the normalized distance of the instances
     * 
     * @param instance1
     *            first instance
     * @param instance2
     *            second instance
     * @return distance
     */
    public double distance(Instance instance1, Instance instance2) {
        double dist = 0;
        for (int i = 0; i < usedAttributes.length; i++) {
            int attr = usedAttributes[i];
            double val1 = 0, val2 = 0;
            if (instance1.isMissing(attr))
                val1 = Double.NaN;
            else
                val1 = instance1.value(attr);
            if (instance2.isMissing(attr))
                val2 = Double.NaN;
            else
                val2 = instance2.value(attr);
            double d = distance(attr, val1, val2);
            dist += d * d;
        }
        return Math.sqrt(dist);
    }

    /**
     * Compute the distance of the values for the given attribute
     * 
     * @param attribute
     *            Index of the attribute
     * @param val1
     *            first value
     * @param val2
     *            second value
     * @return
     */
    public double distance(int attribute, double val1, double val2) {
        if (distNominal!=null && distNominal[attribute] != null) {
            if (!useVDM) {
                // System.out.println("ERROR vdm");
                return val1 != val2 ? 1 : 0;
            }
            if (Double.isNaN(val1) || val1<0)
                val1 = distNominal[attribute].length - 1;
            if (Double.isNaN(val2) || val2<0)
                val2 = distNominal[attribute].length - 1;
            return distNominal[attribute][(int) val1][(int) val2];
        } else if (isUsedAttributes[attribute] == NUMERIC) {
            if ((Double.isNaN(val1)) || (Double.isNaN(val2)))
                return 0.0;// 0.25;
            if (numeric == NO_NUMERIC)
                return val1 - val2;
            else
                return (val1 - val2) / (maxDiff[attribute]);            
        }
//        else if (instances.attribute(attribute).isNominal() && useVDM) {
//            // System.out.println("ERROR2");
//        
//        	int [] sum = new int [distribs[attribute].length] ;
//        	
//        	for (int i=0 ; i< distribs[attribute].length ; i++)
//        	{
//        		for (int j=0 ; j< distribs[attribute][i].length ; j++)
//        		{
//        			sum[i]+=distribs[attribute][i][j] ;
//        		}
//        	}
//        	double diffeence=0 ;
//        	
//        	for (int j=0 ; j< distribs[attribute][(int)val1].length ; j++)
//    		{
//        		diffeence+= Math.abs( distribs[attribute][(int)val1][j]/sum[(int)val1] -
//        		distribs[attribute][(int)val2][j]/sum[(int)val2] );
//    		}            
//            return diffeence ;
//        }
        else if (instances.attribute(attribute).isNominal()) {
            // System.out.println("ERROR2");
            return val1 != val2 ? 1 : 0;
        } else if (instances.attribute(attribute).isNumeric()) {
            // System.out.println("ERROR3");
            double val = val1 - val2;
            return val;
        } else
            return 0;
    }

}