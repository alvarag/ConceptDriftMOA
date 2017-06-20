/*
 * Created on 26.04.2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package utils.indexstructure;

import java.io.Serializable;

/**
 * @author George
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public interface Distance<E> extends Serializable{
	
	public double distance(E o1, E o2);

}
