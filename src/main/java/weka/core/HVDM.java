package weka.core;

/**
 *  Implementation of the Heterogeneous Value Difference Metric
 *  @author Omar Alejandro Mainegra Sarduy (omainegra@uclv.edu.cu)
 */

import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.Instance;

import java.util.*;
import java.io.*;

/**
 * Modifications of the original code from: https://www.dropbox.com/s/s2t2ogaki1x1n4w/Weka.rar?dl=0
 * To use it in MOA.
 * 
 * @author Álvar Arnaiz-González
 * @version 20160522
 */
public class HVDM implements UpdateableDistanceFunction, Serializable, Cloneable{

	private static final long serialVersionUID = -920874830473538978L;

	/** Dataset */
    protected Instances mData = null;
    
    protected int mNumClasses = 0;
    
    protected int mNumAtt = 0;         
    
    protected double [][][] Na = null;
    
    protected double [] mStdDev = null;
        	
    /**
     *  Default Constructor
     *  If you use it, after you must call "setInstances" methods
     */
    
    public HVDM(){
    }
	
    public HVDM(Instances instances){
        mData = instances;
        updateLearning();
    }
    
    public void updateLearning(){
        mNumClasses = mData.numClasses();
        mNumAtt = mData.numAttributes();
                
        Na = new double[mNumAtt][][];
        mStdDev = new double [mNumAtt];
        for (int att = 0; att < mNumAtt; att++){
            if (mData.classIndex() != att){
                if (mData.attribute(att).isNominal()){
                    Na[att] = new double[mData.attribute(att).numValues()][mNumClasses + 1];
                    for (int inst = 0; inst < mData.numInstances(); inst++){
                        Na[att][(int)mData.instance(inst).value(att)][(int)mData.instance(inst).classValue()]++;
                        Na[att][(int)mData.instance(inst).value(att)][mNumClasses]++;
                    }
                }
                else{            
                    double mean = 0.0;
                    for (int inst = 0; inst < mData.numInstances(); inst++)
                        if (!mData.instance(inst).isMissing(att))
                            mean += mData.instance(inst).value(att)/mData.numInstances();            

                    for (int inst = 0; inst < mData.numInstances(); inst++){
                        if (!mData.instance(inst).isMissing(att))
                            mStdDev[att] += Math.pow(mData.instance(inst).value(att) - mean,2)/(mData.numInstances() - 1);            
                    }
                    mStdDev[att] = Math.sqrt(mStdDev[att]);
                }
            }
        }
    }
    
    public String globalInfo() {      
      return "HVDM implements VDM for nominals attributes and normalizes continuous by Standard Deviation";
    }       
    
    public void setInstances(Instances instances){
        mData = instances;
        updateLearning();
    }
    
    public Instances getInstances(){
    	return mData;
    }                

    public Enumeration listOptions() {
        return new Vector().elements();
    }
    
    public void setOptions(String[] options) throws Exception {
    }
  
    public String [] getOptions() {
        return new String[]{""};    	
    } 

    /** 
    *   Calculates the distance (or similarity) between two instances. 
    *   @param first the first instance
    *   @param second the second instance
    *
    *   @return the distance between the two given instances.
    */
    
    public double distance(Instance first, Instance second) throws Exception{
        return distance(first,second,Double.MAX_VALUE);
    }
	
    public double distance(Instance first, Instance second, double cutOffValue) throws Exception{
        if (mData == null)
            throw new Exception("No dataset has been set yet");
//        if (!mData.instance(0).equalHeaders(first) || !mData.instance(0).equalHeaders(second))
        if ((mData.instance(0).numAttributes() != first.numAttributes())
            || (mData.instance(0).classIndex() != first.classIndex())
            || (mData.instance(0).numAttributes() != second.numAttributes())
            || (mData.instance(0).classIndex() != second.classIndex()))
            throw new Exception("Differents types of instances");        
        if (!first.classAttribute().isNominal())
            throw new Exception("Class Attribute is not nominal");                
 			
        double distance = 0; 		
        for (int att = 0; att < mNumAtt; att++)
            if (att != mData.classIndex()){
                double difference = 0; 				 				                
                if (first.isMissing(att) || second.isMissing(att))
                    difference = 1;
                else
                if (mData.attribute(att).isNominal()){                    
                    for (int clazz = 0; clazz < mNumClasses; clazz++)
                        difference += Math.pow(Math.abs(Pauc(att,first.value(att),clazz) - Pauc(att,second.value(att),clazz)),2);
                    difference = Math.sqrt(difference);
                }    
                else
                if (first.attribute(att).isNumeric())
                    if (mStdDev[att] != 0){
                        difference = Math.abs(first.value(att) - second.value(att))/(4*mStdDev[att]);
                    }
                distance += difference*difference;	
            }
 		
            if (distance > cutOffValue)
                distance = Double.MAX_VALUE; 			
            return Math.sqrt(distance);
    }
        
    protected double Pauc(int att, double attValue, double classValue){
        double P = 0;
        if ((attValue >= 0) && (attValue < Na[att].length) && (Na[att][(int)attValue][(int)mNumClasses] != 0.0))
            P = Na[att][(int)attValue][(int)classValue]/Na[att][(int)attValue][mNumClasses];
        return P;
    }
    
    public void postProcessDistances(double distances[]){
    }
    
    public void update(Instance ins) throws Exception{
    }

    public void add(Instance instance) {
		for (int att = 0; att < mData.numAttributes(); att++) {
			if (att != mData.classIndex()) {				
				if (mData.attribute(att).isNumeric()) {					
										
				} else if (mData.attribute(att).isNominal()) {											
					Na[att][(int) instance.value(att)][(int) instance.classValue()]++;
					Na[att][(int) instance.value(att)][mNumClasses]++;
				}
			}
		}
	}

	public void remove(Instance instance) {
		for (int att = 0; att < mData.numAttributes(); att++) {
			if (att != mData.classIndex()) {
				if (mData.attribute(att).isNumeric()) {					
					
				} else if (mData.attribute(att).isNominal()) {	
					if (Na[att][(int) instance.value(att)][(int) instance.classValue()] > 0){						
						Na[att][(int) instance.value(att)][(int) instance.classValue()]--;
					}
					
					if (Na[att][(int) instance.value(att)][mNumClasses] > 0){						
						Na[att][(int) instance.value(att)][mNumClasses]--;
					}					
				}				
			}
		}
	}
}