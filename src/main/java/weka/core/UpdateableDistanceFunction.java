package weka.core;

import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.Instance;

/**
 * Modifications of the original code from:
 * https://www.dropbox.com/s/s2t2ogaki1x1n4w/Weka.rar?dl=0 To use it in MOA.
 * 
 * @author Álvar Arnaiz-González
 * @version 20160522
 */
public interface UpdateableDistanceFunction {// extends DistanceFunction {

	/**
	 * Update the distance function with the information of the newly added
	 * instance
	 */
	public void add(Instance instance);

	/**
	 * Update the distance function with the information of the newly remove
	 * instance
	 */
	public void remove(Instance instance);

	/**
	 * Calculates the distance between two instances. Offers speed up (if the
	 * distance function class in use supports it) in nearest neighbour search
	 * by taking into account the cutOff or maximum distance. Depending on the
	 * distance function class, post processing of the distances by
	 * postProcessDistances(double []) may be required if this function is used.
	 * 
	 * @param first
	 *            the first instance
	 * @param second
	 *            the second instance
	 * @param cutOffValue
	 *            If the distance being calculated becomes larger than
	 *            cutOffValue then the rest of the calculation is discarded.
	 * @return the distance between the two given instances or
	 *         Double.POSITIVE_INFINITY if the distance being calculated becomes
	 *         larger than cutOffValue.
	 */
	public double distance(Instance first, Instance second, double cutOffValue)
			throws Exception;

	/**
	 * Calculates the distance between two instances.
	 * 
	 * @param first
	 *            the first instance
	 * @param second
	 *            the second instance
	 * @return the distance between the two given instances
	 */
	public double distance(Instance first, Instance second) throws Exception;

	/**
	 * Sets the instances.
	 * 
	 * @param insts
	 *            the instances to use
	 */
	public void setInstances(Instances insts);

	/**
	 * returns the instances currently set.
	 * 
	 * @return the current instances
	 */
	public Instances getInstances();

}