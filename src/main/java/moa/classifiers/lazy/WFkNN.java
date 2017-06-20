/*
 * WFkNN.java
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

import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;

/**
 * Abstract class of kNN methods without predefined limited window.
 * <p>
 * Valid options are:
 * <p> 
 * -n number of nearest neighbour <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160523
 */
public abstract class WFkNN extends AbstractClassifier {

	private static final long serialVersionUID = 2695104804821802789L;

	public IntOption kOption = new IntOption("k", 'k',
	                  "The number of neighbors", 10, 1, Integer.MAX_VALUE);

	public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
	        "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use",
	        new String[] {"LinearNN", "KDTree"},
	        new String[] {"Brute force search algorithm for nearest neighbour search.",
	                      "KDTree search algorithm for nearest neighbour search"}, 0);

	protected Instances mWindow;

	protected int mNumClasses = 0;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			mWindow = new Instances(context, 0);
			mWindow.setClassIndex(context.classIndex());
			mNumClasses = context.numClasses();
		} catch (Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

    public double[] getVotesForInstance(Instance inst) {
		double v[] = new double[mNumClasses + 1];
		try {
			NearestNeighbourSearch search;
			if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
				search = new LinearNNSearch(this.mWindow);  
			} else {
				search = new KDTree();
				search.setInstances(this.mWindow);
			}	
			if (this.mWindow.numInstances()>0) {	
				Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.mWindow.numInstances()));
				for(int i = 0; i < neighbours.numInstances(); i++) {
					v[(int)neighbours.instance(i).classValue()]++;
				}
			}
		} catch(Exception e) {
			return new double[inst.numClasses()];
		}
		return v;
    }

	@Override
	public void resetLearningImpl() {
		mWindow = null;
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}
	
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}
}
