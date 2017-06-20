/*
 * Created on 16.03.2004
 *
 * To change the template for this generated file go to
 * Window - Preferences - Java - Code Generation - Code and Comments
 */
package utils.stats;

import java.util.Arrays;


/**
 * This class collect a double stream to compute mean, deviation and quantile values.
 */
public class Distribution {
	
	/**
	 * Normalize an double array with specified quantile value.
	 * @param values
	 * @param quantilePercent
	 * @return normalized values
	 */
	public static double[] normalize(double[] values,double quantilePercent){
		return new Distribution().add(values).norm(values,quantilePercent);
	}
	/**
	 * Normalize an integer array with specified quantile value.
	 * @param values
	 * @param quantilePercent
	 * @return normalized values
	 */
	public static double[] normalize(int[] values,double quantilePercent){
		return new Distribution().add(values).norm(values,quantilePercent);
	}
	
	/** Array of sorted values.*/
	protected double[] val;
	/** Count of values.*/
	protected int count;
	/** Sum of all values.*/
	protected double sum;
	/** Quadratic sum of all values.*/
	protected double sum2;
	
	/**
	 * Initialize an empty Distribution.
	 */
	public Distribution(){
		count=0;
	}
	/**
	 * Add a value to the distribution.
	 * @param value the new value
	 * @return this distribution
	 */
	public Distribution add(double value){
		sum+=value;
		sum2+=value*value;
		if (count==0){
			val=new double[]{value};
			count=1;
			return this;
		}
		double[] old=val;
		val=new double[1+count];
		int pos=Arrays.binarySearch(old,value);
		if (pos<0) pos=-pos-1;
		System.arraycopy(old,0,val,0,pos);
		System.arraycopy(old,pos,val,pos+1,count-pos);
		val[pos]=value;
		count++;
		return this;
	}
	/**
	 * Remove a value of the distribution.
	 * @param value the value to remove.
	 * @return this distribution
	 */
	public Distribution remove(double value){
		int pos=Arrays.binarySearch(val,value);
		if (pos<0) return this;
		sum-=value;
		sum2-=value*value;
		double[] old=val;
		val=new double[count-1];
		System.arraycopy(old,0,val,0,pos);
		System.arraycopy(old,pos+1,val,pos,count-pos-1);
		count--;
		return this;
	}
	/**
	 * Add a array of values to the distribution.
	 * @param values the new values.
	 * @return this distribution
	 */
	public Distribution add(double[] values){
		int len=values.length;
		for (int i=0;i<len;i++){
			sum+=values[i];
			sum2+=values[i]*values[i];
		}
		double[] old=val;
		val=new double[len+count];
		
		if (count>0) System.arraycopy(old,0,val,0,count);
		System.arraycopy(values,0,val,count,len);
		Arrays.sort(val);
		count=count+len;
		return this;
	}
	/**
	 * Add a array of integer values to the distribution.
	 * @param values
	 * @return  this distribution
	 */
	public Distribution add(int[] values){
		int len=values.length;
		double[] old=val;
		val=new double[len+count];
		if (count>0) System.arraycopy(old,0,val,0,count);
		
		for (int i=0;i<len;i++){
			val[i+count]=values[i];
			sum+=values[i];
			sum2+=values[i]*values[i];
		}
		Arrays.sort(val);
		count=count+len;
		return this;
	}
	/**
	 * Get the current quantile value of the distribution
	 * @param percent quantile between 0 and 1
	 * @return the quantile value
	 */
	public double getQuantile(double percent){
		if (percent>1.0) throw new IllegalArgumentException("quantile percent > 1!!!!!");
		int pos=Math.min((int)(count*percent),count-1);
		return val[pos];
	}
	/**
	 * Get the specified value of the sorted data.
	 * @param pos position
	 * @return value
	 */
	public double getValue(int pos){
		if (pos>=count) throw new IllegalArgumentException("quantile percent > 1!!!!!");
		return val[pos];
	}
	/**
	 * Normalize an array of doubles with the given quantile.
	 * @param values
	 * @param quantilePercent
	 * @return normalized values
	 */
	public double[] norm(double[] values,double quantilePercent){
		double one=getQuantile(quantilePercent);
		double[] normed=new double[values.length];
		for (int i=0;i<values.length;i++){
			normed[i]=values[i]/one;
		}
		return normed;
	}
	/**
	 * Normalize an array of integer with the given quantile.
	 * @param values
	 * @param quantilePercent
	 * @return normalized values
	 */
	public double[] norm(int[] values,double quantilePercent){
		double one=getQuantile(quantilePercent);
		double[] normed=new double[values.length];
		for (int i=0;i<values.length;i++){
			normed[i]=values[i]/one;
		}
		return normed;
	}
	/**
	 * Get the mean value of the data.
	 * @return mean value
	 */
	public double getMean(){
		return sum/count;
	}
	/**
	 * Get the minimum value of the data.
	 * @return the minimum
	 */
	public double getMin(){
		return val[0];
	}
	/**
	 * Get the maximum value of the data.
	 * @return the maximum
	 */
	public double getMax(){
		return val[count-1];
	}
	/**
	 * Get the variance of the data.
	 * @return the variance
	 */
	public double getVariance(){
		double m=getMean();
		return sum2/count-m*m;
	}
	/**
	 * Get the standard deviation of the data.
	 * @return the standard deviation
	 */
	public double getStdDev(){
		return Math.sqrt(getVariance());
	}

}
