package utils ;

import java.math.*;
import java.util.Random;

public class IncrementalVariance {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		IncrementalVariance incVariance = new IncrementalVariance() ; 
		int number = 1000 ;
		double [] array = getRandomVector(number) ;
//		for (int i = 0 ;i < number ; i++)
//		{
//			System.out.println(array[i]) ;
//		}
		
		double mean =0 ;
		double variance=0 ;	
		int count = 0 ;
		
/*
		System.out.println("HHHHHHHHHHHHH") ;
		mean = incVariance.CalcMean(0,100,array) ;
		System.out.println(mean) ;
		count =100 ;
		for (int i = 100 ;i < array.length ; i++)
		{
			count++ ;
			mean = incVariance.UpdateMean(mean, array[i], count) ;			
			System.out.println(mean) ;
		}

		System.out.println("HHHHHHHHHHHHH") ;
		mean =0 ;
		count =99 ;
		for (int i = 0 ;i <= array.length-100 ; i++)
		{
			count++ ;
			mean = incVariance.CalcMean(0,count,array) ;			
			System.out.println(mean) ;
		}
*/
		
		
/*				
		System.out.println("HHHHHHHHHHHHH") ;
		mean=0 ;
		
		mean= incVariance.CalcMean(0,100,array) ;
		System.out.println(mean) ;
		
		for (int i = 100 ;i < array.length ; i++)
		{
			mean= incVariance.UpdateMeanOnWindow(mean,array[i-100],array[i],100) ;
			System.out.println(mean) ;
		}
		
		System.out.println("HHHHHHHHHHHHH") ;
		mean=0 ;				
		for (int i = 0 ;i <= array.length-100 ; i++)
		{
			mean=incVariance.CalcMean(i,100,array) ;
			System.out.println(mean) ;
		}
*/
		
		
		
//		System.out.println("HHHHHHHHHHHHH") ;
//		variance=0 ;				
//		for (int i = 0 ;i <= array.length-100 ; i++)
//		{
//			variance=incVariance.CalcVariance(i,100,array) ;
//			System.out.println(variance) ;
//		}
		
/*			
		System.out.println("HHHHHHHHHHHHH") ;
		variance=0 ;
		mean = 0 ;
		mean = incVariance.CalcMean(0,100,array) ;
		variance=incVariance.CalcVariance(0,100,array) ;
		System.out.println(variance+","+mean) ;
		
		count =100 ;
		for (int i = 100 ;i < array.length ; i++)
		{
			count++ ;
			double meanCurr = incVariance.UpdateMean(mean, array[i], count) ;				
			variance=incVariance.UpdateVariance(variance, mean, meanCurr, array[i], count) ;
			
			mean = meanCurr ;
			System.out.println(variance +","+mean) ;
		}


		System.out.println("HHHHHHHHHHHHH") ;
		mean=0 ;
		variance=0 ;
		count =99 ;
		for (int i = 0 ;i <= array.length-100 ; i++)
		{
			count++ ;
			mean=incVariance.CalcMean(0,count,array) ;
			variance=incVariance.CalcVariance(0,count,array) ;			
			System.out.println(variance +","+mean) ;
		}
*/
		
		
		System.out.println("HHHHHHHHHHHHH") ;
		mean=0 ;
		variance=0 ;
		
		mean= incVariance.CalcMean(0,100,array) ;
		variance=incVariance.CalcVariance(0,100,array) ;
		System.out.println(variance+","+mean) ;
		
		for (int i = 100 ;i <array.length  ; i++) //
		{
			double meanCurr= incVariance.UpdateMeanOnWindow(mean,array[i-100],array[i],100) ;
			variance= incVariance.UpdateVarianceOnWindow(variance, mean, meanCurr, array[i-100],array[i], 100) ;
			
			mean = meanCurr ;
			System.out.println(variance +","+mean) ;
		}
		
		System.out.println("HHHHHHHHHHHHH") ;
		
		mean=0 ;
		variance=0 ;
		
		double [] temp= new double[100] ; 
		for (int i = 0 ; i <100 ; i++)
			temp[i] = array[i] ;
		
		incVariance = new IncrementalVariance(temp) ;
		System.out.println(incVariance.Variance +","+incVariance.Mean) ;
		for (int i = 100 ;i <array.length  ; i++) //
		{
			incVariance.UpdateMeanVarianceOnWindow(array[i]) ;			
			
			System.out.println(incVariance.Variance +","+incVariance.Mean) ;
		}
		
		System.out.println("HHHHHHHHHHHHH") ;
		
		mean=0 ;				
		for (int i = 0 ;i <=array.length-100 ; i++) //
		{
			mean=incVariance.CalcMean(i,100,array) ;
			variance=incVariance.CalcVariance(i,100,array) ;			
			System.out.println(variance +","+mean) ;
		}
	}	
	
	public double CalcMean(int start, int length , double [] array)
	{
		double mean= 0 ;
		double sum = 0 ;
		for (int i = start ; i <start + length ; i++)
			sum+= array[i] ;
		mean= sum/length ;
		return mean ;		
	}
	
	public double UpdateMean(double mean , double newVal, int count)
	{	
		mean = mean + (newVal - mean )/count ;
		return mean ;
	}
	
	
	

	public double CalcVariance(int start, int length , double [] array)
	{
		double mean= CalcMean(start, length, array) ; ;
		double sum = 0 ;
		for (int i = start ; i <start + length ; i++)
			sum+= Math.pow(array[i] - mean,2)  ;
		double variance = sum/(length-1);
		return variance ;		
	}
	
	
	public double UpdateVariance(double variance , double meanPrev ,double meanCurr,  double newVal, int count)
	{	
		double temp = variance * (count-2) + (newVal - meanPrev) *  (newVal - meanCurr) ;
		variance = temp/(count-1);
		return variance ;
	}
	
	public double UpdateMeanOnWindow(double mean ,double oldVal, double newVal,int count)
	{		
		mean = mean + (newVal - oldVal )/count ;	
		return mean ;
	}
	
	public double UpdateVarianceOnWindow(double variance ,double meanPrev ,double meanCurr, double oldVal, double newVal,int count)
	{	
		double temp= (newVal - oldVal) *  (newVal + oldVal - meanPrev - meanCurr) ;
		variance = variance + temp/(count-1);	
		return variance ;
	}
	
	public void UpdateMeanVarianceOnWindow(double newVal)
	{	
		double oldVal = array[index] ;
		double meanCurr=UpdateMeanOnWindow(Mean, oldVal, newVal, size) ;
		
		double varianceCurrent = UpdateVarianceOnWindow(Variance, Mean, meanCurr, oldVal, newVal, size) ;
		addValue(newVal) ;
		
		Mean = meanCurr ;
		Variance = varianceCurrent ;
	}
	public static double [] getRandomVector(int number)
	{
		double [] Result = new double[number] ;
		Random rand = new Random(number) ;
		for (int i = 0 ;i < number ; i++)
		{
			Result[i] = rand.nextInt(rand.nextInt(999)+1) ;
		}		
		return Result ;
	}
	
	double [] array = null ;
	int index=0 ;
	int size=0 ;
	
	public double Mean= 0 ;
	public double Variance= 0 ;
	
	public IncrementalVariance()
	{
		
	}
	
	public IncrementalVariance(double [] arrayV)
	{
		array= arrayV ;
		size= arrayV.length ;
		index = 0 ;
		
		Mean= CalcMean(0,array.length,array) ;
		Variance=CalcVariance(0,array.length,array) ;
	}
	
	public void addValue (double item)
	{
		array[index] = item ;
		if ( index==size-1 )
			index= 0 ;
		else 
			index++ ;
	}
}
