package utils;

public class Utils {
    
    /** The small deviation allowed in double comparisons */
    public static double SMALL = 1e-6;
    
    /**
     * Tests if a is equal to b.
     *
     * @param a a double
     * @param b a double
     */
    public static /*@pure@*/ boolean eq(double a, double b){
      
      return (a - b < SMALL) && (b - a < SMALL); 
    }
    
    
    /**
     * Sorts a given array of integers in ascending order and returns an 
     * array of integers with the positions of the elements of the original 
     * array in the sorted array. The sort is stable. (Equal elements remain
     * in their original order.)
     *
     * @param array this array is not changed by the method!
     * @return an array of integers with the positions in the sorted
     * array.
     */
    public static /*@pure@*/ int[] sort(int [] array) {

      int [] index = new int[array.length];
      int [] newIndex = new int[array.length];
      int [] helpIndex;
      int numEqual;
      
      for (int i = 0; i < index.length; i++) {
        index[i] = i;
      }
      quickSort(array, index, 0, array.length - 1);

      // Make sort stable
      int i = 0;
      while (i < index.length) {
        numEqual = 1;
        for (int j = i + 1; ((j < index.length)
               && (array[index[i]] == array[index[j]]));
       j++) {
    numEqual++;
        }
        if (numEqual > 1) {
    helpIndex = new int[numEqual];
    for (int j = 0; j < numEqual; j++) {
      helpIndex[j] = i + j;
    }
    quickSort(index, helpIndex, 0, numEqual - 1);
    for (int j = 0; j < numEqual; j++) {
      newIndex[i + j] = index[helpIndex[j]];
    }
    i += numEqual;
        } else {
    newIndex[i] = index[i];
    i++;
        }
      }
      return newIndex;
    }

    /**
     * Sorts a given array of doubles in ascending order and returns an
     * array of integers with the positions of the elements of the
     * original array in the sorted array. NOTE THESE CHANGES: the sort
     * is no longer stable and it doesn't use safe floating-point
     * comparisons anymore. Occurrences of Double.NaN are treated as 
     * Double.MAX_VALUE
     *
     * @param array this array is not changed by the method!
     * @return an array of integers with the positions in the sorted
     * array.  
     */
    public static /*@pure@*/ int[] sort(/*@non_null@*/ double [] array) {

      int [] index = new int[array.length];
      array = (double [])array.clone();
      for (int i = 0; i < index.length; i++) {
        index[i] = i;
        if (Double.isNaN(array[i])) {
          array[i] = Double.MAX_VALUE;
        }
      }
      quickSort(array, index, 0, array.length - 1);
      return index;
    }

    /**
     * Sorts a given array of doubles in ascending order and returns an 
     * array of integers with the positions of the elements of the original 
     * array in the sorted array. The sort is stable (Equal elements remain
     * in their original order.) Occurrences of Double.NaN are treated as 
     * Double.MAX_VALUE
     *
     * @param array this array is not changed by the method!
     * @return an array of integers with the positions in the sorted
     * array.
     */
    public static /*@pure@*/ int[] stableSort(double [] array){

      int [] index = new int[array.length];
      int [] newIndex = new int[array.length];
      int [] helpIndex;
      int numEqual;
      
      array = (double [])array.clone();
      for (int i = 0; i < index.length; i++) {
        index[i] = i;
        if (Double.isNaN(array[i])) {
          array[i] = Double.MAX_VALUE;
        }
      }
      quickSort(array,index,0,array.length-1);

      // Make sort stable

      int i = 0;
      while (i < index.length) {
        numEqual = 1;
        for (int j = i+1; ((j < index.length) && Utils.eq(array[index[i]],
                            array[index[j]])); j++)
    numEqual++;
        if (numEqual > 1) {
    helpIndex = new int[numEqual];
    for (int j = 0; j < numEqual; j++)
      helpIndex[j] = i+j;
    quickSort(index, helpIndex, 0, numEqual-1);
    for (int j = 0; j < numEqual; j++) 
      newIndex[i+j] = index[helpIndex[j]];
    i += numEqual;
        } else {
    newIndex[i] = index[i];
    i++;
        }
      }

      return newIndex;
    }
    
    
    /**
     * Implements quicksort according to Manber's "Introduction to
     * Algorithms".
     *
     * @param array the array of doubles to be sorted
     * @param index the index into the array of doubles
     * @param left the first index of the subset to be sorted
     * @param right the last index of the subset to be sorted
     */
    //@ requires 0 <= first && first <= right && right < array.length;
    //@ requires (\forall int i; 0 <= i && i < index.length; 0 <= index[i] && index[i] < array.length);
    //@ requires array != index;
    //  assignable index;
    private static void quickSort(/*@non_null@*/ double[] array, /*@non_null@*/ int[] index, 
                                  int left, int right) {

      if (left < right) {
        int middle = partition(array, index, left, right);
        quickSort(array, index, left, middle);
        quickSort(array, index, middle + 1, right);
      }
    }
    
    /**
     * Implements quicksort according to Manber's "Introduction to
     * Algorithms".
     *
     * @param array the array of integers to be sorted
     * @param index the index into the array of integers
     * @param left the first index of the subset to be sorted
     * @param right the last index of the subset to be sorted
     */
    //@ requires 0 <= first && first <= right && right < array.length;
    //@ requires (\forall int i; 0 <= i && i < index.length; 0 <= index[i] && index[i] < array.length);
    //@ requires array != index;
    //  assignable index;
    private static void quickSort(/*@non_null@*/ int[] array, /*@non_null@*/  int[] index, 
                                  int left, int right) {

      if (left < right) {
        int middle = partition(array, index, left, right);
        quickSort(array, index, left, middle);
        quickSort(array, index, middle + 1, right);
      }
    }
    
    
    
    /**
     * Partitions the instances around a pivot. Used by quicksort and
     * kthSmallestValue.
     *
     * @param array the array of doubles to be sorted
     * @param index the index into the array of doubles
     * @param left the first index of the subset 
     * @param right the last index of the subset 
     *
     * @return the index of the middle element
     */
    private static int partition(double[] array, int[] index, int l, int r) {
      
      double pivot = array[index[(l + r) / 2]];
      int help;

      while (l < r) {
        while ((array[index[l]] < pivot) && (l < r)) {
          l++;
        }
        while ((array[index[r]] > pivot) && (l < r)) {
          r--;
        }
        if (l < r) {
          help = index[l];
          index[l] = index[r];
          index[r] = help;
          l++;
          r--;
        }
      }
      if ((l == r) && (array[index[r]] > pivot)) {
        r--;
      } 

      return r;
    }

    /**
     * Partitions the instances around a pivot. Used by quicksort and
     * kthSmallestValue.
     *
     * @param array the array of integers to be sorted
     * @param index the index into the array of integers
     * @param left the first index of the subset 
     * @param right the last index of the subset 
     *
     * @return the index of the middle element
     */
    private static int partition(int[] array, int[] index, int l, int r) {
      
      double pivot = array[index[(l + r) / 2]];
      int help;

      while (l < r) {
        while ((array[index[l]] < pivot) && (l < r)) {
          l++;
        }
        while ((array[index[r]] > pivot) && (l < r)) {
          r--;
        }
        if (l < r) {
          help = index[l];
          index[l] = index[r];
          index[r] = help;
          l++;
          r--;
        }
      }
      if ((l == r) && (array[index[r]] > pivot)) {
        r--;
      } 

      return r;
    }

}
