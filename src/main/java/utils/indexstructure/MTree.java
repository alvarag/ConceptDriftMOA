/*
 * Created on 26.04.2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package utils.indexstructure;

import java.io.Serializable;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Stack;

import weka.core.Utils;
import xxl.core.collections.Lists;
import xxl.core.cursors.mappers.Mapper;
import xxl.core.functions.AbstractFunction;

/**
 * @author George
 * 
 * TODO To change the template for this generated type comment go to Window -
 * Preferences - Java - Code Style - Code Templates
 */
public class MTree<E> extends AbstractCollection<E> implements Cloneable, Serializable {

    /**
     * Hyperplane split strategy
     */
    public static final int LINEAR_HYPERPLANE_SPLIT = 0;

    public static final int LINEAR_BALANCED_SPLIT = 1;

    /**
     * Hyperplane split strategy
     */
    public static final int HYPERPLANE_SPLIT = 2;

    /**
     * Balanced split strategy
     */
    public static final int BALANCED_SPLIT = 3;

    public class Sphere implements Cloneable, Iterable, Serializable {
        /**
         * The center of the sphere.
         */
        public Object center;

        /**
         * The radius of the sphere.
         */
        public double radius;

        /** List of subspheres */
        protected List<Sphere> childs;

        /** Level of this Sphere-Node */
        protected int level;

        protected int size;

        /**
         * The distance from the center of this sphere to the center of the
         * sphere of the parent node in the MTree.
         */
        protected double distanceToParent = -1;

        public Sphere(Object element) {
            this(element, 0, null, 0, 1);
        }

        public Sphere(Object center, double radius, List childs, int level,
                int size) {
            this.center = center;
            this.radius = radius;
            this.childs = childs;
            this.level = level;
            this.size = size;
        }

        public Sphere(Object center, int level) {
            this(center, 0, new ArrayList(), level, 0);
        }

        public Sphere(Sphere sp) {
            this(sp.center, sp.radius, new ArrayList(), sp.level + 1, 0);
            childs.add(sp);
            size+=sp.size;
        }

        public Sphere(Object center, double radius, List childs, int level) {
            this.center = center;
            this.radius = radius;
            this.childs = childs;
            this.level = level;
            size = 0;
            for (Sphere sp : this.childs) {
                size += sp.size;
            }
        }

        public Sphere clone(AbstractFunction map) {
            Sphere clone = null;
            try {
                clone = (Sphere) super.clone();
            } catch (CloneNotSupportedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
                return null;
            }
            clone.center = map.invoke(center);
            if (childs != null) {
                clone.childs = new ArrayList<Sphere>(childs.size());
                for (Sphere s : childs) {
                    clone.childs.add(s.clone(map));
                }
            }
            return clone;
        }

        public Object clone() {
            return clone(identity);
        }

        public Iterator<E> iterator() {
            if (level == 1) {
                return new Iterator() {
                    int pos = 0;

                    public boolean hasNext() {
                        return pos < childs.size();
                    }

                    public Object next() {
                        if (hasNext()) {
                            return childs.get(pos++).center;
                        } else
                            throw new NoSuchElementException();
                    }

                    public void remove() {
                        throw new UnsupportedOperationException();
                    }
                };
            } else {
                return new Iterator() {
                    int pos = -1;

                    Iterator sub;

                    public boolean hasNext() {
                        if (sub != null && sub.hasNext())
                            return true;
                        if (pos >= childs.size())
                            return false;
                        while (++pos < childs.size()) {
                            sub = childs.get(pos).iterator();
                            if (sub.hasNext())
                                return true;
                        }
                        return false;
                    }

                    public Object next() {
                        if (hasNext()) {
                            return sub.next();
                        } else
                            throw new NoSuchElementException();
                    }

                    public void remove() {
                        throw new UnsupportedOperationException();
                    }
                };
            }
        }

        public double radius() {
            return radius;
        }

        public Object center() {
            return center;
        }

        public void updateDistance() {
            if (level > 0) {
                double maxRad = 0;
                for (Sphere sp : childs) {
                    sp.updateDistance();
                    sp.distanceToParent = distance.distance(center, sp.center);
                    maxRad = Math.max(maxRad, sp.distanceToParent + sp.radius);
                }
                radius = maxRad;
            }

        }

        public Sphere chooseSphere(Object object) {
            if (level == 0)
                return null;
            Iterator it = childs.iterator();
            double minDist = Double.MAX_VALUE;
            double minErw = 0;
            Sphere minSphere = null;
            if (!it.hasNext())
                return null;
            while (it.hasNext()) {
                Sphere sp = (Sphere) it.next();
                double dist = MTree.this.distance.distance(sp.center, object);
                if (dist < minDist
                        || (dist == minDist && minErw > dist - sp.radius)) {
                    minDist = dist;
                    minErw = dist - sp.radius;
                    minSphere = sp;
                }
            }
            return minSphere;
        }

        public Sphere chooseSphere(Sphere sphere) {
            if (level == 0)
                return null;
            Iterator it = childs.iterator();
            double minDist = Double.MAX_VALUE;
            double minErw = 0;
            Sphere minSphere = null;
            if (!it.hasNext())
                return null;
            while (it.hasNext()) {
                Sphere sp = (Sphere) it.next();
                double dist = MTree.this.distance.distance(sp.center,
                        sphere.center);
                if (dist < minDist
                        || (dist == minDist && minErw > dist - sp.radius
                                + sphere.radius)) {
                    minDist = dist;
                    minErw = dist - sp.radius + sphere.radius;
                    minSphere = sp;
                }
            }
            return minSphere;
        }

        public int remove(List rem, int nr, DoubleFunction func) {
            if (level > 1) {
                int sum = 0;
                for (int i = 0; i < childs.size(); i++) {

                    Sphere sp = childs.get(i);

                    boolean ismax = Math.abs(sp.distanceToParent + sp.radius
                            - radius) < 0.0001;
                    int c = sp.remove(rem, (int) Math.floor(nr*sp.size
                            / (double)size), func);
                    if (c>nr)
                        c+=0;
                    nr -= c;
                    size-=c;
                    sum += c;
                    if (ismax)
                        recomputeRadius();
                    if (sp.distanceToParent + sp.radius > radius) {
                        radius = sp.distanceToParent + sp.radius;
                    }
                    if (sp.childs.size() < minCapacity && childs.size() > 1) {
                        childs.remove(i);
                        recomputeRadius();
                        for (Sphere chsp : sp.childs) {
                            size-=chsp.size;
                            add(chsp);
                        }
                        i--;
                    }
                    if (nr<=0) return sum;
                }
                return sum;
            } else {
                if (nr >= childs.size()) {
                    int n = childs.size();
                    for (Sphere sp : childs) {
                        rem.add(sp.center);
                    }
                    childs.clear();
                    radius = 0;
                    size=0;
                    return n;
                }
                double[] val = new double[childs.size()];
                for (int i = 0; i < childs.size(); i++) {
                    val[i] = func.invoke(childs.get(i).center);
                }
                int[] sort = Utils.sort(val);
                ArrayList<Sphere> l = new ArrayList<Sphere>(nr);
                for (int i = 0; i < nr; i++) {
                    rem.add(childs.get(sort[i]).center);
                    l.add(childs.get(sort[i]));
                }
                childs.removeAll(l);
                size-=nr;
                recomputeRadius();
                return nr;
            }

        }

        public boolean remove(Object ob) {
            double dist=MTree.this.distance.distance(ob, center);
            if (dist - radius > 0.00001)
                return false;
            else {
                if (level > 1) {
                    for (Iterator<Sphere> it = childs.iterator(); it.hasNext();) {

                        Sphere sp = it.next();
                        // if (sp.childs.size() < minCapacity) {
                        // System.out.println("error");
                        // }
                        boolean ismax = Math.abs(sp.distanceToParent
                                + sp.radius - radius) < 0.0001;
                        if (sp.remove(ob)) {
                            size--;
                            if (ismax)
                                recomputeRadius();
                            if (sp.distanceToParent + sp.radius > radius) {
                                radius = sp.distanceToParent + sp.radius;
                            }
                            if (sp.childs.size() < minCapacity
                                    && childs.size() > 1) {
                                it.remove();
                                // if (childs.contains(sp)) {
                                // System.out.println("error");
                                // }
                                recomputeRadius();
                                for (Sphere chsp : sp.childs) {
                                    size-=chsp.size;
                                    add(chsp);
                                }
                            }
                            return true;
                        }
                    }
                } else {
                    for (Iterator<Sphere> it = childs.iterator(); it.hasNext();) {
                        Sphere sp = it.next();
                        if (sp.center == ob) {
                            size--;
                            boolean ismax = Math.abs(sp.distanceToParent
                                    - radius) < 0.0001;
                            it.remove();
                            if (ismax)
                                recomputeRadius();
                            return true;
                        }
                    }
                }
                return false;
            }
        }

        public void add(Sphere addSp) {
            if (level == addSp.level + 1) {
                if (addSp.level == 0
                        || (addSp.childs != null && addSp.childs.size() >= minCapacity)) {
                    childs.add(addSp);
                    size+=addSp.size;
                    addSp.distanceToParent = distance.distance(addSp.center,
                            center);
                    if (radius < addSp.distanceToParent + addSp.radius)
                        radius = addSp.distanceToParent + addSp.radius;
                } else {
                    if (addSp.childs != null) {
                        for (Sphere sp : addSp.childs) {
                            size-=sp.size;
                            add(sp);
                        }
                    }
                }
            } else {
                Sphere sp = chooseSphere(addSp);
                if (sp == null) {
                    sp = chooseSphere(addSp);
                }
                double oldRad = radius;
                boolean ismax = Math.abs(sp.distanceToParent + sp.radius
                        - radius) < 0.0001;
                // checkRadius(this);
                sp.add(addSp);
                size+=addSp.size;
                
                // checkRadius(sp);
                if (radius < sp.distanceToParent + sp.radius) {
                    radius = sp.distanceToParent + sp.radius;
                    ismax = true;
                }
                if (sp.radius + sp.distanceToParent < oldRad && ismax) {
                    recomputeRadius();
                }
                if (sp.childs.size() > maxCapacity) {
                    split(sp, this);
                    if (ismax)
                        recomputeRadius();
                }
            }
        }

        public void recomputeCenter() {
            double[][] distances = new double[childs.size()][childs.size()];
            int i_min = 0;
            double minValue = Double.MAX_VALUE;
            for (int i = 0; i < childs.size(); i++) {
                double maxValue = 0;
                double d = 0;
                Sphere c0 = childs.get(i);
                for (int j = 0; j < i; j++) {
                    Sphere c1 = childs.get(j);
                    d = distances[i][j] + c1.radius;
                    if (d > maxValue)
                        maxValue = d;
                }
                d = c0.radius;
                if (d > maxValue)
                    maxValue = d;
                for (int j = i + 1; j < childs.size(); j++) {

                    Sphere c1 = childs.get(j);
                    d = (distances[j][i] = distances[i][j] = distance.distance(
                            c0.center, c1.center))
                            + c1.radius;
                    if (d > maxValue)
                        maxValue = d;
                }
                if (i == 0 || maxValue < minValue) {
                    i_min = i;
                    minValue = maxValue;
                }
            }
            center = childs.get(i_min).center;
            for (int i = 0; i < childs.size(); i++) {
                Sphere c0 = childs.get(i);
                c0.distanceToParent = distances[i_min][i];
            }
            radius = minValue;
        }

        public void recomputeRadius() {
            if (level == 0) {
                radius = 0;
            } else {
                double maxRad = 0;
                for (Sphere sp : (List<Sphere>) childs) {
                    maxRad = Math.max(maxRad, sp.distanceToParent + sp.radius);
                }
                radius = maxRad;
            }
        }

        public String toString() {
            return center.toString() + " :" + radius;
        }

//        public void check() {
//            if (level > 0) {
//                int s = 0;
//                double rad = 0;
//                    
//                for (Sphere sp : childs) {
//                    sp.check();
//                    double pd = distance.distance(center, sp.center);
//                    if (Math.abs(pd - sp.distanceToParent) > 0.000001)
//                        throw new Error();
//                    if (sp.level>0&&(sp.childs.size()<minCapacity||sp.childs.size()>maxCapacity))
//                        throw new Error();
//                    s += sp.size;
//                    rad = Math.max(rad, sp.distanceToParent + sp.radius);
//                }
//                if (Math.abs(rad-radius) > 0.000001)
//                    throw new Error();
//                if (s!=size)
//                    throw new Error();
//                
//            }
//        }

    }

    

    protected class ObjectDistance implements Serializable{
        Sphere sphere;

        double d;
    }
    
    public class AllIterator implements Iterator<E>, Serializable{
        E next = null;

        Sphere nextSphere = null;

        Sphere lastSphere = null;

        LinkedList<Sphere> list = new LinkedList();

        Stack<Sphere> path = new Stack();

        public boolean hasNext() {
            if (nextSphere == null) {
                list.add(root);
            }
            lastSphere = null;
            if (next != null)
                return true;
            while (list.size() > 0) {
                Sphere s = (Sphere) list.removeFirst();
                if (s.level == 0) {
                    next = (E)s.center;
                    nextSphere = s;
                    if (list.size() > 0) {
                        int l = list.getFirst().level;
                        int l2 = path.peek().level;
                        while (l >= l2) {
                            path.pop();
                            l2++;
                        }
                    }
                    return true;
                } else {
                    list.addAll(0, s.childs);
                    path.push(s);
                }
            }
            return false;
        }

        public E next() {
            if (next == null) {
                hasNext();
            }
            E n = next;
            next = null;
            lastSphere = nextSphere;
            return n;
        }

        public void remove() {
            MTree.this.remove(lastSphere.center);
        }
    
    }

    protected DoubleFunction<E> random = new DoubleFunction<E>() {
        public double invoke(E e) {
            return Math.random();
        }
    };

    protected Distance distance;

    protected int size;

    protected int minCapacity;

    protected int maxCapacity;

    protected Sphere root;

    /**
     * A prototype function simply returns its arguments (identity-function).
     */
    protected AbstractFunction<E, E> identity = new AbstractFunction<E, E>() {

        /**
         * Returns the argument itself (identity-function).
         * 
         * @param arguments
         *            the argument of the function.
         * @return the <code>argument</code> is returned.
         */
        public final E invoke(List< ? extends E> arguments) {
            if (arguments.size() == 1)
                return arguments.get(0);
            else
                throw new IllegalArgumentException(
                        "only one argument allowed!!!");
        }

        public final E invoke(E object) {
            return object;
        }
    };

    /**
     * The split strategy used in this <tt>MTree</tt>.
     */
    private int splitMode = LINEAR_HYPERPLANE_SPLIT;//LINEAR_BALANCED_SPLIT;

    protected MTree() {
        init(null, 0, 0);
    }

    public MTree(Distance<E> distance, int minCapacity, int maxCapacity) {
        init(distance, minCapacity, maxCapacity);
    }

    public void init(Distance<E> distance, int minCapacity, int maxCapacity) {
        this.distance = distance;
        this.minCapacity = minCapacity;
        this.maxCapacity = maxCapacity;
        size = 0;
        root = null;
    }

    /**
     * @param splitMode
     *            The splitMode to set.
     */
    public void setSplitMode(int splitMode) {
        this.splitMode = splitMode;
    }

    /**
     * @return Returns the splitMode.
     */
    public int getSplitMode() {
        return splitMode;
    }

    public void updateDistance(Distance<E> distance) {
        this.distance = distance;
        if (root != null) {
            root.updateDistance();
        }
    }

    /** Hilfsvariablen zur split-Zeitmessung. Summe aller splitaufrufe */
    public double time = 0;

    /** Anzahl der split-Aufrufe */
    public int count = 0;

    protected void split(Sphere sp, Sphere parent) {
        // split bei Blättern nicht möglich
        if (sp.level > 0) {
            long start = System.nanoTime();
            List<Sphere> childs = sp.childs;
            int size = childs.size();
            Sphere sp1 = null;
            Sphere sp2 = null;

            int pos, s1;
            int sel;
            int p0;
            int p1;
            int n0;
            int n1;
            double maxVal, minmax;
            double[] dist, dist2, dist1, dist0;
            Object o1;

            Sphere first0, first1, last0, last1;

            switch (getSplitMode()) {
            case LINEAR_HYPERPLANE_SPLIT:
                // Suche entferntesten Unterknoten vom Zentrum als Grundlage für
                // einen der neuen Knoten
                maxVal = -1;
                int ind1 = -1;
                for (int i = 0; i < size; i++) {
                    Sphere ch = childs.get(i);
                    if (maxVal < ch.distanceToParent) {
                        maxVal = ch.distanceToParent;
                        sp1 = ch;
                        ind1 = i;
                    }
                }
                maxVal = -1;
                // Berechne Abstand vom entferntesten zu allen anderen
                // Unterknoten
                dist = new double[size];
                int ind2 = -1;
                for (int i = 0; i < size; i++) {
                    Sphere ch = childs.get(i);
                    if (ch==sp1){
                        dist[i]=0;
                        continue;
                    }
                    double d1 = distance.distance(sp1.center, ch.center);
                    dist[i] = d1;
                    if (d1 > maxVal) {
                        maxVal = d1;
                        sp2 = ch;
                        ind2 = i;
                    }
                }

                // sortiere Unterknoten nach Abständen

                // nimm den gegenüberliegenden entferntesten für den 2. neuen
                // Knoten
                o1 = sp2.center;
                // berechne auch Abstände aller Unterknoten zu diesem und
                // sortiere ebenfalls
                dist2 = new double[size];
                for (int i = 0; i < size; i++) {
                    Object c = childs.get(i).center;
                    dist2[i] = distance.distance(o1, c);
                }

                // berechne Grösse der neuen Knoten
                s1 = (int) Math.ceil(size / 2.0);
                int maxC = size - minCapacity;
                int insertToAdd = -1;
                double max0 = -1,
                max1 = -1;

                // Subknoten-Listen für die neuen Knoten
                ArrayList<Sphere> li0;
                ArrayList<Sphere> li1;
                li0 = new ArrayList(size / 2);
                li1 = new ArrayList(size / 2);

                // erste subknoten die sicher in den jeweils neuen Knoten liegen
                first0 = sp1;
                first1 = sp2;
                // in den jeweils neuen Knoten entfernteste Subknoten zu
                // first0/first1
                last0 = null;
                last1 = null;
                // Distanzen innerhalb der neuen Knoten zu dem schon
                // feststehenen Knoten (first0/first1)
                dist0 = new double[maxC];
                dist1 = new double[maxC];
                li0.add(sp1);
                childs.set(ind1, null);
                li1.add(sp2);
                childs.set(ind2, null);
                for (int i = 0; i < size; i++) {
                    Sphere ch = childs.get(i);
                    if (ch == null)
                        continue;
                    int add = -1;
                    double d0 = dist[i];
                    double d1 = dist2[i];

                    if (insertToAdd >= 0) {
                        add = insertToAdd;
                    } else if (max0 >= d0 && max1 >= d1) {
                        if (li0.size() <= li1.size()) {
                            add = 0;
                        } else {
                            add = 1;
                        }
                    } else if (max0 >= d0) {
                        add = 0;
                    } else if (max1 >= d1) {
                        add = 1;
                    } else if (d0 < d1) {
                        add = 0;
                    } else {
                        add = 1;
                    }
                    if (add == 0) {
                        dist0[li0.size()] = dist[i];
                        li0.add(ch);

                        if (max0 < d0) {
                            max0 = d0;
                            last0 = ch;
                        }
                        if (li0.size() >= maxC)
                            insertToAdd = 1;
                    } else {
                        dist1[li1.size()] = dist2[i];
                        li1.add(ch);

                        if (max1 < d1) {
                            max1 = d1;
                            last1 = ch;
                        }
                        if (li1.size() >= maxC)
                            insertToAdd = 0;
                    }
                }

                pos = -1;
                minmax = Double.MAX_VALUE;
                o1 = last0.center;
                s1 = li0.size();
                // Ermittlung des neuen Zentrums für Knoten 0
                // minimierung des maximalen Abstandes zu den 2 entferntesten
                // Knoten first0, last0
                for (int i = 0; i < s1; i++) {
                    Object c = li0.get(i).center;
                    double d1 = distance.distance(o1, c);
                    double d2 = dist0[i];
                    if (Math.max(d1, d2) < minmax) {
                        minmax = Math.max(d1, d2);
                        pos = i;
                    }
                }
                // neuen Knoten erzeugen und subknoten einfügen
                sp1 = new Sphere(li0.get(pos).center, sp.level);
                for (int i = 0; i < s1; i++) {
                    Sphere ch = li0.get(i);
                    sp1.add(ch);
                }

                // Zentrum auch für 2.Knoten ermitteln
                pos = -1;
                minmax = Double.MAX_VALUE;
                o1 = last1.center;
                s1 = li1.size();
                for (int i = 0; i < s1; i++) {
                    Object c = li1.get(i).center;
                    double d1 = distance.distance(o1, c);
                    double d2 = dist1[i];
                    if (Math.max(d1, d2) < minmax) {
                        minmax = Math.max(d1, d2);
                        pos = i;
                    }
                }
                // erzeugen und einfügen
                sp2 = new Sphere(li1.get(pos).center, sp.level);
                for (int i = 0; i < s1; i++) {
                    Sphere ch = li1.get(i);
                    sp2.add(ch);
                }

                break;
            case LINEAR_BALANCED_SPLIT:
                // Suche entferntesten Unterknoten vom Zentrum als Grundlage für
                // einen der neuen Knoten
                maxVal = -1;
                for (Sphere ch : childs) {
                    if (maxVal < ch.distanceToParent) {
                        maxVal = ch.distanceToParent;
                        sp1 = ch;
                    }
                }
                // Berechne Abstand vom entferntesten zu allen anderen
                // Unterknoten
                dist = new double[size];
                for (int i = 0; i < size; i++) {
                    Sphere ch = childs.get(i);
                    if (ch==sp1){
                        dist[i]=0;
                        continue;
                    }
                    double d1 = distance.distance(sp1.center, ch.center);
                    dist[i] = d1;

                }
                // sortiere Unterknoten nach Abständen
                int[] sortDist = Utils.sort(dist);

                // nimm den gegenüberliegenden entferntesten für den 2. neuen
                // Knoten
                o1 = childs.get(sortDist[size - 1]).center;
                // berechne auch Abstände aller Unterknoten zu diesem und
                // sortiere ebenfalls
                dist2 = new double[size];
                for (int i = 0; i < size; i++) {
                    Object c = childs.get(i).center;
                    dist2[i] = distance.distance(o1, c);
                }
                int[] sortDist2 = Utils.sort(dist2);

                // berechne Grösse der neuen Knoten
                s1 = (int) Math.ceil(size / 2.0);
                // Subknoten-Listen für die neuen Knoten
                li0 = new ArrayList<Sphere>(s1);
                li1 = new ArrayList<Sphere>(s1);
                sel = 0;
                p0 = 0;
                p1 = 0;
                n0 = 0;
                n1 = 0;
                // erste subknoten die sicher in den jeweils neuen Knoten liegen
                first0 = sp1;
                first1 = childs.get(sortDist[size - 1]);
                // in den jeweils neuen Knoten entfernteste Subknoten zu
                // first0/first1
                last0 = null;
                last1 = null;
                // Distanzen innerhalb der neuen Knoten zu dem schon
                // feststehenen Knoten (first0/first1)
                dist0 = new double[s1];
                dist1 = new double[s1];
                // balancierte abwechselnde Aufteilung der Subknoten in die
                // neuen Knoten
                while (sel < size) {
                    while (childs.get(sortDist[p0]) == null)
                        p0++;
                    li0.add(last0 = childs.set(sortDist[p0], null));
                    dist0[n0] = dist[sortDist[p0]];
                    n0++;
                    sel++;
                    if (sel < size) {
                        while (childs.get(sortDist2[p1]) == null)
                            p1++;
                        li1.add(last1 = childs.set(sortDist2[p1], null));
                        dist1[n1] = dist2[sortDist2[p1]];
                        n1++;
                        sel++;
                    }
                }

                pos = -1;
                minmax = Double.MAX_VALUE;
                o1 = last0.center;
                s1 = li0.size();
                // Ermittlung des neuen Zentrums für Knoten 0
                // minimierung des maximalen Abstandes zu den 2 entferntesten
                // Knoten first0, last0
                for (int i = 0; i < s1; i++) {
                    Object c = li0.get(i).center;
                    double d1 = distance.distance(o1, c);
                    double d2 = dist0[i];
                    if (Math.max(d1, d2) < minmax) {
                        minmax = Math.max(d1, d2);
                        pos = i;
                    }
                }
                // neuen Knoten erzeugen und subknoten einfügen
                sp1 = new Sphere(li0.get(pos).center, sp.level);
                for (int i = 0; i < s1; i++) {
                    Sphere ch = li0.get(i);
                    sp1.add(ch);
                }

                // Zentrum auch für 2.Knoten ermitteln
                pos = -1;
                minmax = Double.MAX_VALUE;
                o1 = last1.center;
                s1 = li1.size();
                for (int i = 0; i < s1; i++) {
                    Object c = li1.get(i).center;
                    double d1 = distance.distance(o1, c);
                    double d2 = dist1[i];
                    if (Math.max(d1, d2) < minmax) {
                        minmax = Math.max(d1, d2);
                        pos = i;
                    }
                }
                // erzeugen und einfügen
                sp2 = new Sphere(li1.get(pos).center, sp.level);
                for (int i = 0; i < s1; i++) {
                    Sphere ch = li1.get(i);
                    sp2.add(ch);
                }

                break;
            case HYPERPLANE_SPLIT:
            case BALANCED_SPLIT:
                maxVal = -1;
                int maxi = -1;
                int maxj = -1;
                for (int i = 0; i < childs.size() - 1; i++) {
                    Object c1 = childs.get(i).center;
                    for (int j = i + 1; j < childs.size(); j++) {
                        Object c2 = childs.get(j).center;
                        double d = distance.distance(c1, c2);
                        if (d > maxVal) {
                            maxVal = d;
                            maxi = i;
                            maxj = j;
                        }
                    }
                }
                final Sphere sphere0 = childs.get(maxi),
                sphere1 = childs.get(maxj);
                ArrayList<Sphere>[] collections = new ArrayList[] {
                        new ArrayList(size / 2), new ArrayList(size / 2) };
                ArrayList<Sphere> insertTo = null;
                switch (getSplitMode()) {
                case HYPERPLANE_SPLIT:
                    maxC = size - minCapacity;
                    insertTo = null;

                    for (Sphere ch : childs) {
                        if (insertTo != null) {
                            insertTo.add(ch);
                            continue;
                        }
                        double d0 = distance
                                .distance(ch.center, sphere0.center);
                        double d1 = distance
                                .distance(ch.center, sphere1.center);
                        if (d0 < d1) {
                            collections[0].add(ch);
                            if (collections[0].size() >= maxC)
                                insertTo = collections[1];
                        } else {
                            collections[1].add(ch);
                            if (collections[1].size() >= maxC)
                                insertTo = collections[0];
                        }
                    }
                    break;
                case BALANCED_SPLIT: /* balanced distribution */
                    ArrayList<Sphere> NNSphere0 = (ArrayList<Sphere>) childs;
                    Lists.quickSort(NNSphere0,
                            getDistanceBasedComparator(sphere0));
                    ArrayList<Sphere> NNSphere1 = (ArrayList<Sphere>) ((ArrayList<Sphere>) childs)
                            .clone();
                    Lists.quickSort(NNSphere1,
                            getDistanceBasedComparator(sphere1));

                    while (!(NNSphere0.isEmpty() && NNSphere1.isEmpty())) {
                        Sphere next;
                        if (!NNSphere0.isEmpty()) {
                            next = NNSphere0.get(0);
                            collections[0].add(next);
                            NNSphere0.remove(0);
                            NNSphere1.remove(next);
                        }
                        if (!NNSphere1.isEmpty()) {
                            next = NNSphere1.get(0);
                            collections[1].add(next);
                            NNSphere1.remove(0);
                            NNSphere0.remove(next);
                        }
                    }
                    break;
                }
                sp1 = new Sphere(collections[0].get(0), 0, collections[0],
                        sp.level);
                sp1.recomputeCenter();
                sp2 = new Sphere(collections[1].get(0), 0, collections[1],
                        sp.level);
                sp2.recomputeCenter();
            }
            // alten Knoten löschen und neue hinzufügen
            parent.childs.remove(sp);//TODO richtigen löschen!!!!
            parent.size-=sp.size;
            parent.add(sp1);
            parent.add(sp2);
            long end = System.nanoTime();
            time += (end - start);
            count++;
        }
    }
    
    public class DBComparator  implements Comparator,Serializable{
        Sphere referenceSphere;
        DBComparator(Sphere referenceSphere){
            this.referenceSphere=referenceSphere;
        }
        
        public int compare(Object o1, Object o2) {
            double dist1 = distance.distance(((Sphere) o1).center,
                    referenceSphere.center);
            double dist2 = distance.distance(((Sphere) o2).center,
                    referenceSphere.center);
            return dist1 < dist2 ? 1 : dist1 > dist2 ? -1 : 0;
        }
    }

    /**
     * Returns a comparator comparing spheres with respect to their distance to
     * the given <tt>referenceSphere</tt>.
     * 
     * @param referenceSphere
     *            the rererence for the comparison of the comparator
     * @return a comparator returning 1, 0 or -1 if its first parameter has
     *         smaller, equal or greater distance to <tt>referenceSphere</tt>
     *         than the second one
     */
    protected Comparator getDistanceBasedComparator(final Sphere referenceSphere) {
        return new DBComparator(referenceSphere);
            
    }
    
    protected class NNComparator implements Comparator, Serializable{
        public int compare(Object arg0, Object arg1) {
            double d0 = ((ObjectDistance) arg0).d;
            double d1 = ((ObjectDistance) arg1).d;
            if (d0 < d1)
                return -1;
            if (d0 == d1)
                return 0;
            return 1;
        }
    }

    protected Comparator nnComparator = new NNComparator();

    public Iterator<E> getNearestNeighbours(final E instance) {
        final DoubleFunction<Sphere> dFunc = new DoubleFunction<Sphere>() {
            public double invoke(Sphere s) {
                return distance.distance(s.center, instance) - s.radius;
            }
        };
        return query(dFunc);
    }
    
    public class QueryIterator implements Iterator<E>, Serializable{
        Object next = null;

        ObjectDistance nextEntry = null;

        ObjectDistance lastEntry = null;
        
        Heap heap;
        
        AbstractFunction mapFunc;
        
        public QueryIterator(Heap heap,AbstractFunction mapFunc){
            this.heap=heap;
            this.mapFunc=mapFunc;
        }

        public boolean hasNext() {
            lastEntry = null;
            if (next != null)
                return true;
            while (heap.size() > 0) {
                ObjectDistance o = (ObjectDistance) heap.next();
                if (o==null) continue;
                Sphere s = o.sphere;
                if (s.level == 0) {
                    next = s.center;
                    nextEntry = o;
                    return true;
                } else {
                    heap
                            .insertAll(new Mapper(mapFunc, s.childs
                                    .iterator()));
                }
            }
            return false;
        }

        public E next() {
            if (next==null) hasNext();
            Object n = next;
            next = null;
            lastEntry = nextEntry;
            return (E)n;
        }

        public void remove() {
            MTree.this.remove(lastEntry.sphere.center);
        }
    }

    public Iterator<E> query(final DoubleFunction<Sphere> dFunc) {
        final AbstractFunction mapFunc = new AbstractFunction() {
            public Object invoke(Object o) {
                if (o == null)
                    System.out.println("null");
                Sphere s = (Sphere) o;
                ObjectDistance n = new ObjectDistance();
                n.sphere = s;
                n.d = dFunc.invoke(s);
                return n;
            }
        };
        if (root == null)
            return null;
        final Heap heap = new Heap(nnComparator);
        Object n = mapFunc.invoke(root);
        heap.insert(n);
        return new QueryIterator(heap,mapFunc);
        
    }

    public E getNearestNeighbour(E instance) {
        Iterator<E> it = getNearestNeighbours(instance);
        if (it.hasNext())
            return it.next();
        return null;
    }
    
   

    public Iterator<E> iterator() {
        return new AllIterator();
            

    }

    public int size() {
        return size;
    }

    public void clear() {
        root = null;

    }

    public boolean isEmpty() {

        return root == null;
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.util.Collection#add(java.lang.Object)
     */
    public boolean add(E arg0) {
        if (root == null) {
            Sphere s = new Sphere(arg0);
            root = new Sphere(s);
            s.distanceToParent = 0;
        } else if (root.level == 0) {
            Sphere r2 = new Sphere(root.center, 1);
            r2.add(root);
            Sphere s = new Sphere(arg0);
            r2.add(s);
            root = r2;
        } else {
            Sphere sp = new Sphere(arg0);
            double oldRad = root.radius;
            // checkRadius(root);
            root.add(sp);
            if (root.childs.size() > maxCapacity) {
                Sphere r2 = new Sphere(root.center, root.level + 1);
                r2.size=root.size;
                split(root, r2);
                // vielleicht neues Zentrum von root optimieren
                root = r2;
                root.recomputeCenter();
                // checkRadius(root);
            }
            //checkRadius(root);
        }
        size++;
        
        return true;
    }

    public MTree clone(AbstractFunction map) {
        MTree<E> clone = null;
        try {
            clone = (MTree) super.clone();
        } catch (CloneNotSupportedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        }
        clone.size = size;
        clone.root = root.clone(map);
        return clone;
    }

    public Object clone() {
        return clone(identity);
    }

    /**
     * @param arg0
     * @param path
     */
    protected void path(Object arg0, Sphere sphere, Stack path) {
        while (sphere.level > 0) {
            Sphere subsphere = sphere.chooseSphere(arg0);
        }

    }

    /*
     * (non-Javadoc)
     * 
     * @see java.util.Collection#remove(java.lang.Object)
     */
    public boolean remove(Object arg0) {
        if (root == null)
            return false;
        if (root.level == 0) {
            if (root.center == arg0) {
                root = null;
                size = 0;
                return true;
            } else
                return false;
        }
        if (root.remove(arg0)) {
            size--;
            if (root.childs.size() == 1) {
                root = root.childs.get(0);
                root.distanceToParent = 0;
            }
            //checkRadius(root);
            return true;
        }
        return false;
    }

    public List remove(int nr) {
        return remove(nr, random);
    }

    public List remove(int nr, DoubleFunction<E> func) {
        
        if (root != null) {
            List rem = new ArrayList(nr);
            int c = root.remove(rem, nr, func);
            size -= c;
            return rem;
        }
        return null;
    }

//    public void checkRadius() {
//        List<Sphere> all = new ArrayList();
//        all.add(root);
//        while (!all.isEmpty()) {
//            Sphere sp = all.remove(all.size() - 1);
//            checkRadius(sp);
//            sp.check();
//            if (sp.level > 1)
//                all.addAll(sp.childs);
//        }
//    }
//
//    public void checkRadius(Sphere sp) {
//        double rad = sp.radius;
//        sp.recomputeRadius();
//        if (Math.abs(sp.radius - rad) > 0.00001)
//            System.out.println("Error");
//    }

}
