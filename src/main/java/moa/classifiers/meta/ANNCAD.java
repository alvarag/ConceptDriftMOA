/*
 * ANNCAD.java
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
package moa.classifiers.meta;

import java.util.Random;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.lazy.ANNCADTree;
import moa.core.Measurement;
import moa.options.ClassOption;

/**
 * Implementation of Adaptive NN Classification Algorithm for 
 * Data-streams (ANNCAD). Presented in:
 * Law, Y. N., & Zaniolo, C. (2005). An adaptive nearest neighbor 
 * classification algorithm for data streams. In Knowledge Discovery 
 * in Databases: PKDD 2005 (pp. 108-120). Springer Berlin Heidelberg.
 * <p>
 * Valid options are:
 * <p>
 * -l base learner (must be always ANNCADTree) <br>
 * -z ensemble size <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160520
 */
public class ANNCAD extends AbstractClassifier {

	private static final long serialVersionUID = 9120097552361574641L;

	public ClassOption mBaseLearnerOption = new ClassOption("baseLearner", 'l',
	         "ANNCAD.", Classifier.class, "lazy.ANNCADTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 'z',
             "The number of models to ANNCAD.", 2, 1, Integer.MAX_VALUE);

    protected Classifier[] mEnsemble;

	@Override
	public boolean isRandomizable() {

		return true;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		double[] tmpV, v = new double[inst.numClasses()];
		
		for (int i = 0; i < mEnsemble.length; i++) {
			tmpV = mEnsemble[i].getVotesForInstance(inst);
			
			for (int j = 0; j < tmpV.length; j++)
				v[j] += tmpV[j];
		}
		
		return v;
	}

	@Override
    public void resetLearningImpl() {
		Random random = new Random(randomSeed);
		
        mEnsemble = new ANNCADTree[ensembleSizeOption.getValue()];
        
        Classifier baseLearner = (Classifier)getPreparedClassOption(this.mBaseLearnerOption);
        baseLearner.resetLearning();
        
        for (int i = 0; i < mEnsemble.length; i++) {
            mEnsemble[i] = (ANNCADTree)baseLearner.copy();
            mEnsemble[i].setRandomSeed(random.nextInt());
        }
    }

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		for (int i = 0; i < mEnsemble.length; i++)
            mEnsemble[i].trainOnInstance(inst);
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
        return "Adaptive NN Classification Algorithm for Data-streams (ANNCAD)";
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.mEnsemble.clone();
    }
}
