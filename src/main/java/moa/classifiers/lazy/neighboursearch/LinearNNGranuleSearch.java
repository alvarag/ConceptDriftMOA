/*
 * LinearNNGranuleSearch.java
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
package moa.classifiers.lazy.neighboursearch;

import java.util.ArrayList;

import moa.classifiers.lazy.oiGRLVQ.Granule;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

/**
 * Auxiliary class for search in a list of granules.
 * 
 * @author Álvar Arnaiz-González
 * @version 20160526
 */
public class LinearNNGranuleSearch extends LinearNNSearch {
	
	private static final long serialVersionUID = 5495533221653831963L;

	/**
	 * Granules where the search will be performed.
	 */
	private ArrayList<Granule> mGranules;
	
	/**
	 * Default constructor of a granules' search.
	 * The dataset will not be taken into account.
	 * 
	 * @param granules Arraylist of granules.
	 * @param data Dataset with the information of attributes and class.
	 */
	public LinearNNGranuleSearch (ArrayList<Granule> granules, Instances data) {
		super(data);
		mGranules = granules;
	}
	
	/**
	 * Returns the nearest neighbour of target.
	 * 
	 * @param target The instance to find the k nearest neighbours for.
	 * @return the k nearest neighbors
	 * @throws Exception if the neighbours could not be found.
	 */
	public Instance nearestNeighbour(Instance target) throws Exception {
		return (kNearestNeighbours(target, 1)).instance(0);
	}

	/**
	 * Returns k nearest instances in the current neighbourhood to the
	 * supplied instance.
	 * Extracted from LinearNNSearch of MOA.
	 * 
	 * @param target The instance to find the k nearest neighbours for.
	 * @param kNN  The number of nearest neighbours to find.
	 * @return the k nearest neighbors
	 * @throws Exception if the neighbours could not be found.
	 */
	public Instances kNearestNeighbours(Instance target, int kNN) throws Exception {
		MyHeap heap = new MyHeap(kNN);
		double distance;
		int firstkNN = 0;
		for (int i = 0; i < mGranules.size(); i++) {
			if (target == mGranules.get(i)) // for hold-one-out
													// cross-validation
				continue;
			if (firstkNN < kNN) {
				distance = m_DistanceFunction.distance(target,
						mGranules.get(i).instance(), Double.POSITIVE_INFINITY);
				if (distance == 0.0 && m_SkipIdentical)
					if (i < mGranules.size() - 1)
						continue;
					else
						heap.put(i, distance);
				heap.put(i, distance);
				firstkNN++;
			} else {
				MyHeapElement temp = heap.peek();
				distance = m_DistanceFunction.distance(target,
						mGranules.get(i).instance(), temp.distance);
				if (distance == 0.0 && m_SkipIdentical)
					continue;
				if (distance < temp.distance) {
					heap.putBySubstitute(i, distance);
				} else if (distance == temp.distance) {
					heap.putKthNearest(i, distance);
				}

			}
		}

		Instances neighbours = new Instances(target.dataset(),
		            (heap.size() + heap.noOfKthNearest()));
		m_Distances = new double[heap.size() + heap.noOfKthNearest()];
		int[] indices = new int[heap.size() + heap.noOfKthNearest()];
		int i = 1;
		MyHeapElement h;
		while (heap.noOfKthNearest() > 0) {
			h = heap.getKthNearest();
			indices[indices.length - i] = h.index;
			m_Distances[indices.length - i] = h.distance;
			i++;
		}
		while (heap.size() > 0) {
			h = heap.get();
			indices[indices.length - i] = h.index;
			m_Distances[indices.length - i] = h.distance;
			i++;
		}

		m_DistanceFunction.postProcessDistances(m_Distances);

		for (int k = 0; k < indices.length; k++)
			neighbours.add(mGranules.get(indices[k]).instance());

		return neighbours;
	}
}
