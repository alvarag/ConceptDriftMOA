/*
 * TWF.java
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

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

/**
 * Implementation of Time-Weighted Forgetting (TWF).
 * <p>
 * Valid options are:
 * <p> 
 * -f forgetting rate <br>
 * -t threshold <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160523
 */
public class TWF extends WFkNN {

	private static final long serialVersionUID = 5470746585240664971L;

	public FloatOption mForgettingRate = new FloatOption("forgettingRate", 'f',
	                                          "Forgetting rate", 0.996, 0, 1);

	public FloatOption mThreshold = new FloatOption("threshold", 't', "Threshold", 
	                                     0.2, 0, 1);

	private int mLimitOption;
	
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (mWindow == null)
			mWindow = new Instances(inst.dataset());
		
		if (mLimitOption <= mWindow.numInstances())
			mWindow.delete(0);
		
		mWindow.add(inst);
	}

	public String getPurposeString() {
		return "Time-Weighted Forgetting (TWF).";
	}

	@Override
	public void setModelContext(InstancesHeader context) {
		mWindow = new Instances(context,0);
		mWindow.setClassIndex(context.classIndex());
		mNumClasses = context.classAttribute().numValues();
		mLimitOption = (int)Math.round(Math.log(mThreshold.getValue()) / 
		                     Math.log(mForgettingRate.getValue()));
	}
}
