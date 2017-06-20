package utils.indexstructure;

import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * This class provides the heap datastructure. The <tt>heapify</tt> operation
 * and the constructors that initialize the heap with an array of objects run in
 * amortized constant time, that is, adding n elements requires O(n) time.
 * <p>
 */

public class Heap extends AbstractCollection {

    protected static Collection resize;
    static {
        resize = new ArrayList();
        for (int i = 0; i < 10; i++) {
            resize.add(null);
        }
    }

    /**
     * The array is internally used to store the elements of the heap.
     */
    protected Object[] array;

    /**
     * An <tt>int</tt> field storing the index of the last element of the heap
     * in the array (size=last+1).
     */
    protected int last;

    /**
     * The comparator to determine the order of the heap. More exactly, there
     * can be three different cases when two elements <tt>o1</tt> and
     * <tt>o2</tt> are inserted into the heap
     * <ul>
     * <dl>
     * <dt>
     * <li><tt>comparator.compare(o1, o2) < 0</tt> :</dt>
     * <dd>the heap returns <tt>o1</tt> prior to returning <tt>o2</tt>.</dd>
     * <dt>
     * <li><tt>comparator.compare(o1, o2) == 0</tt> :</dt>
     * <dd>when inserting equal elements (determined by the used comparator),
     * there is no guarantee which one will be returned first.
     * <dt>
     * <li><tt>comparator.compare(o1, o2) > 0</tt> :</dt>
     * <dd>the heap returns <tt>o2</tt> prior to returning <tt>o1</tt>.</dd>
     * </dl>
     * </ul>
     */
    protected Comparator comparator;

    /**
     * Constructs a heap containing the elements of the specified array that
     * returns them according to the order induced by the specified comparator.
     * The specified array has two different functions. First the heap depends
     * on this array and is not able to contain more elements than the array is
     * able to. Second it is used to initialize the heap. The field
     * <tt>array</tt> is set to the specified array, the field <tt>last</tt>
     * is set to the specified size - 1 and the field <tt>comparator</tt> is
     * set to the specified comparator. After initializing the fields the
     * heapify method is called.
     * 
     * @param array
     *            the object array that is used to store the heap and initialize
     *            the internally used array.
     * @param size
     *            the number of elements of the specified array which should be
     *            used to initialize the heap.
     * @param comparator
     *            the comparator to determine the order of the heap.
     * @throws IllegalArgumentException
     *             if the specified size argument is negative, or if it is
     *             greater than the length of the specified array.
     */
    public Heap(List array, int size, Comparator comparator)
            throws IllegalArgumentException {
        if (array.size() < size || size < 0)
            throw new IllegalArgumentException();
        this.array = array.toArray();
        this.last = size - 1;
        this.comparator = comparator;
        heapify();
    }

    /**
     * Constructs an empty heap and uses the <i>natural ordering</i> (Object
     * has to implement Comparable) of the elements to order them when they are
     * inserted. This constructor is equivalent to the call of
     * <code>new Heap(0)</code>.
     */
    public Heap() {
        this(0);
    }

    /**
     * Constructs an empty heap and uses the specified comparator to order
     * elements when inserted. This constructor is equivalent to the call of
     * <code>new Heap(0, comparator)</code>.
     * 
     * @param comparator
     *            the comparator to determine the order of the heap.
     */
    public Heap(Comparator comparator) {
        this(0, comparator);
    }

    /**
     * Constructs a heap containing the elements of the specified array that
     * returns them according to the order induced by the specified comparator.
     * This constructor is equivalent to the call of
     * <code>Heap(array, array.size(), comparator)</code>.
     * 
     * @param array
     *            the object array that is used to store the heap and initialize
     *            the internally used array.
     * @param comparator
     *            the comparator to determine the order of the heap.
     */
    public Heap(List array, Comparator comparator) {
        this(array, array.size(), comparator);
    }

    /**
     * Constructs an empty heap with a capacity of size elements and uses the
     * specified comparator to order elements when inserted. This constructor is
     * equivalent to the call of
     * <code>Heap(new ArrayList(size), 0, comparator)</code>.
     * 
     * @param size
     *            the maximal number of elements the heap is able to contain.
     * @param comparator
     *            the comparator to determine the order of the heap.
     */
    public Heap(int size, Comparator comparator) {
        this(new ArrayList(size), 0, comparator);
    }

    /**
     * Constructs a heap containing the elements of the specified array that
     * returns them according to their <i>natural ordering</i>.
     * 
     * @param array
     *            the object array that is used to store the heap and initialize
     *            the internally used array.
     * @param size
     *            the number of elements of the specified array which should be
     *            used to initialize the heap.
     */
    public Heap(List array, int size) {
        this(array, size, new Comparator() {
            public int compare(Object o1, Object o2) {
                return ((Comparable) o1).compareTo(o2);
            }
        });
    }

    /**
     * Constructs a heap containing the elements of the specified array that
     * returns them according to their <i>natural ordering</i>. This
     * constructor is equivalent to the call of
     * <code>Heap(array, array.size())</code>.
     * 
     * @param array
     *            the object array that is used to store the heap and initialize
     *            the internally used array.
     */
    public Heap(List array) {
        this(array, array.size());
    }

    /**
     * Constructs an empty heap with an initial capacity of size elements and
     * uses the <i>natural ordering</i> of the elements to order them when they
     * are inserted. This constructor is equivalent to the call of
     * <code>Heap(new Object[size], size)</code>.
     * 
     * @param size
     *            the maximal number of elements the heap is able to contain.
     */
    public Heap(int size) {
        this(new ArrayList(size), size);
    }

    /**
     * Ensure a size of count. Add null-values if the list is to small.
     * 
     * @param count
     *            new Size of List
     */
    private void ensure(int count) {
        if (count >= array.length) {
            int l = array.length;
            while (count >= l) {
                l += 10;
            }
            Object[] arr = new Object[l];
            System.arraycopy(array, 0, arr, 0, array.length);
            array = arr;
        }
    }

    /**
     * Computes the heap for the first (<tt>last + 1</tt>) elements of the
     * internally used array in O(n) time.
     */
    private void heapify() {
        if (last > 0) {
            for (int i = (last - 1) >> 1; i > 0; i--) {
                Object top = array[i >> 1];
                bubbleUp(array[i], sinkIn(i));
                array[i >> 1] = top;
            }
            insert(replace(array[last--]));
        }
    }

    /**
     * Inserts the specified object into the heap and overwrites the element at
     * the index i of the internally used array without damaging the structure
     * of the heap. The specified object is inserted into the path from the root
     * of the heap to the element with the index i and the whole path beyond the
     * inserted element is <i>shifted</i> one level down. This method only
     * works fine when the following prerequisites are valid
     * <ul>
     * <li><tt>0 <= i <= last</tt>
     * <li><tt>comparator.compare(array[0], object) <= 0</tt>
     * </ul>
     * 
     * @param object
     *            the object to insert into the heap.
     * @param i
     *            an index of the internally used array. The specified object is
     *            inserted into the path from the root of the heap to the
     *            element with the index i.
     */
    private void bubbleUp(Object object, int i) {
        // prerequisite: 0 <= i <= last && array[0] <= object
        while (comparator.compare(object, array[i >> 1]) < 0 ) {
            array[i] = array[i >>= 1];
        }
        array[i] = object;
    }

    /**
     * Removes the element at the index i/2 of the internally used array without
     * damaging the structure of the heap. The whole path from the element at
     * the index i of the internally used array to the bottom level of the heap
     * is shifted one level up. This method only works fine when the following
     * prerequisite is valid
     * <ul>
     * <li><tt>1 <= i <= last</tt>
     * </ul>
     * 
     * @param i
     *            an index of the internally used array. The object at the index
     *            i/2 of the internally used array is removed.
     * @return the index of the last element in the path that is shifted one
     *         level up.
     */
//    private int sinkIn(int i) {
//        // prerequisite: 1 <= i <= last
//        array[i >> 1] = array[i];
//        int i2;
//        while ((i2=i << 1) < last){
//            if (comparator.compare(array[i], array[i + 1]) < 0) i2++;
//            array[i] = array[i2];
//            i=i2;
//        }
//        return i >> 1;
//    }
    
    protected int sinkIn(int i) {
        // prerequisite: 1 <= i <= last
        array[i>>1] = array[i];
        while ((i <<= 1) < last)
            array[(comparator.compare(array[i], array[i+1]) < 0 ? i : i++)>>1] = array[i];
        return i>>1;
    }

    /**
     * Removes all of the elements from this heap. The heap will be empty after
     * this call returns so that <tt>size() == 0</tt>.
     */
    public void clear() {
        last = -1;
        array = new Object[10];
    }

    /**
     * Returns the number of elements in this heap. If this heap contains more
     * than <tt>Integer.MAX_VALUE</tt> elements, returns
     * <tt>Integer.MAX_VALUE</tt>.
     * 
     * @return the number of elements in this queue.
     */
    public int size() {
        return last + 1;
    }

    /**
     * Inserts the specified element into this heap and restores the structure
     * of the heap if necessary.
     * 
     * @param object
     *            element to be inserted into this heap.
     */
    public void insert(Object object) {
        ensure(last + 2);
        if (++last > 0) {
            if (comparator.compare(object, array[0]) < 0) {
                Object top = array[0];
                array[0]= object;
                object = top;
            }
            bubbleUp(object, last);
        } else
            array[0]= object;
    }

    /**
     * Inserts all of the elements in the specified iterator into this heap and
     * restores the structure of the heap if necessary. The behavior of this
     * operation is unspecified if the specified iterator is modified while the
     * operation is in progress.
     * 
     * @param objects
     *            iterator whose elements are to be inserted into this heap.
     */
    public void insertAll(Iterator objects) {
        if (size() > 0)
            while (objects.hasNext())
                insert(objects.next());
        else {
            while (objects.hasNext()) {
                ensure(last + 2);
                array[++last]= objects.next();
            }
            heapify();
        }
    }

    /**
     * Inserts all of the elements in the specified array into this heap and
     * restores the structure of the heap if necessary.
     * 
     * @param objects
     *            array whose elements are to be inserted into this heap.
     */
    public void insertAll(Object[] objects) {
        int len = objects.length;
        ensure(last + 1 + len);
        if (size() > 0)
            for (int i = 0; i < len; i++) {
                insert(objects[i]);
            }
        else {
            for (int i = 0; i < len; i++) {
                array[++last]= objects[i];
            }
            heapify();
        }
    }

    /**
     * Returns the <i>next</i> element in the heap without removing it. The
     * <i>next</i> element of the heap is determined by the comparator.
     * 
     * @return the <i>next</i> element in the heap.
     * @throws NoSuchElementException
     *             heap has no more elements.
     * @throws UnsupportedOperationException
     *             if the <tt>peek</tt> operation is not supported by this
     *             heap.
     */
    public Object peek() throws NoSuchElementException,
            UnsupportedOperationException {
        if (size() == 0)
            throw new NoSuchElementException();
        return array[0];
    }

    /**
     * Returns the <i>next</i> element in the heap and remove it. The <i>next</i>
     * element of the heap is determined by the comparator.
     * 
     * @return the <i>next</i> element in the heap.
     * @throws NoSuchElementException
     *             heap has no more elements.
     */
    public Object next() throws NoSuchElementException {
        if (last >= 0) {
            Object minimum = array[0];
            if (last > 0) {
                bubbleUp(array[last], sinkIn(1));
                array[last]=null;
            }
            last--;
            return minimum;
        } else
            throw new NoSuchElementException();
    }

    /**
     * Replaces the next element in the heap with the specified object and
     * restore the structure of the heap if necessary. More exactly, the
     * specified object is inserted into the heap and the next element of the
     * heap is overwritten. Thereafter the structure of the heap is restored and
     * the overwritten element is returned. This method only works fine when the
     * following prerequisite is valid
     * <ul>
     * <li><tt>comparator.compare(array[0], object) <= 0</tt>
     * </ul>
     * 
     * @param object
     *            element to be inserted into this heap.
     * @return the <i>next</i> element in the heap.
     */
    public Object replace(Object object) throws NoSuchElementException {
        // prerequisite: array[0] <= object
        if (last >= 0) {
            Object minimum = array[0];
            if (last > 0 && comparator.compare(object, array[1]) > 0) {
                int up = sinkIn(1);
                if (2 * up == last)
                    array[up] = array[up = last];
                bubbleUp(object, up);
            } else
                array[0] = object;
            return minimum;
        } else
            throw new NoSuchElementException();
    }

    /**
     * The main method contains an examples how to use this Heap. It can also be
     * used to test the functionality of a Heap.
     * 
     * @param args
     *            array of <tt>String</tt> arguments. It can be used to submit
     *            parameters when the main method is called.
     */
    public static void main(String[] args) {

        // create a comparator that compares the objects by comparing
        // their Integers
        Comparator comparator = new Comparator() {
            public int compare(Object o1, Object o2) {
                return ((Integer) ((Object[]) o1)[0]).intValue()
                        - ((Integer) ((Object[]) o2)[0]).intValue();
            }
        };

        Heap heap = new Heap(comparator);

        for (int i = 0; i < 100; i++)
            heap.insert(new Object[] {
                    new Integer((int) (Math.random() * 100)), i + "." });
        // print the elements of the heap
        while (heap.size() > 0) {
            Object[] o = (Object[]) heap.next();
            System.out.println("Integer = " + o[0] + " & String = " + o[1]);
        }
        System.out.println();
    }

    @Override
    public Iterator iterator() {
        return new Iterator() {
            int pos = 0;

            public boolean hasNext() {
                return pos <= last;
            }

            public Object next() {
                return array[pos];
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }

        };
    }
}