package utils.stats;

import java.util.Comparator;
import java.util.Iterator;
import java.util.TreeSet;

public class ClassifyTimeTTest {
    
    protected static class ClassifyData{
        double correct;
        double time;
        
        public ClassifyData(double time,double correct){
            this.correct=correct;
            this.time=time;
        }
        
        public String toString(){
            return time+":"+correct;
        }
    }

    protected Comparator<ClassifyData> comp = new Comparator<ClassifyData>() {
        public int compare(ClassifyData ob1, ClassifyData ob2) {
            if (ob1.time < ob2.time)
                return 1;
            else if (ob1.time > ob2.time)
                return -1;
            else
                return ob2.hashCode() - ob1.hashCode();
        }
    };

    protected double pmin = Double.MAX_VALUE;

    protected double smin = Double.MAX_VALUE;

    protected int histlen = 100;

    protected double lastSum = 0;

    protected int warning;

    protected TreeSet<ClassifyData> lastCor;

    protected double p, s;

    protected boolean full;

    protected double warnLevel;

    protected double signLevel;

    public ClassifyTimeTTest(double warnLevel, double signLevel, int histlen) {
        this.warnLevel = warnLevel;
        this.signLevel = signLevel;
        this.histlen = histlen;
        init();
    }
    
    public ClassifyTimeTTest(ClassifyTimeTTest test){
        this.warnLevel = test.warnLevel;
        this.signLevel = test.signLevel;
        this.histlen = test.histlen;
        init();
        addClassify(test.iterator());
    }

    public void init() {
        full = false;
        lastCor = new TreeSet<ClassifyData>(comp);
        lastSum = 0;
        pmin = Double.MAX_VALUE;
        smin = Double.MAX_VALUE;
        warning = 0;
    }

    public boolean isSignificant() {
        return full && p + s > pmin + signLevel * smin;
    }

    public double[] getCurrentValues() {
        return new double[] { p, s };
    }

    public double[] getMinValues() {
        return new double[] { pmin, smin };
    }

    public int getWarning() {
        return warning;
    }

    public void resetStats() {
        pmin = Double.MAX_VALUE;
        smin = Double.MAX_VALUE;
//        classifyCount = 0;
//        full=false;
//        lastCor.clear();
        warning = 0;
    }

    public void addOptimize(double fak) {
        pmin = pmin+ fak;
        smin = Math.sqrt(pmin * (1 - pmin) / histlen);;
    }

    public double getProbDiff() {
        int len = histlen / 3;
        int sumNew = 0;
        Iterator<ClassifyData> it = lastCor.iterator();
        for (int i = 0; i < len; i++) {
            sumNew += it.next().correct;
        }

        return (1.0 - pmin) - (sumNew / (double) len);
    }

    public int addClassify(Iterator<ClassifyData> it) {
        int last = 0;
        while (it.hasNext()) {
            ClassifyData next = it.next();
            last = addClassify(next);
        }
        return last;
    }
    
    public void setLevel(double warnLevel, double signLevel, int histlen) {
        this.warnLevel = warnLevel;
        this.signLevel = signLevel;
        this.histlen = histlen;
        if (lastCor.size()<histlen) full=false;
        while (lastCor.size()>histlen){
            ClassifyData last = lastCor.last();
            lastSum -= last.correct;
            lastCor.remove(last);
        }
    }

    public int addClassify(double correct, double time) {
        return addClassify(new ClassifyData( time, correct ));
    }

    protected int addClassify(ClassifyData classify) {
        if (lastCor.contains(classify)) return 0;
        int ss = lastCor.size();
        lastCor.add(classify);
        if (lastCor.size() <= ss)
            System.out.println("Err");
        lastSum += classify.correct;
        while (lastCor.size()>histlen){
            ClassifyData last = lastCor.last();
            lastSum -= last.correct;
            lastCor.remove(last);
        }
        if (!full && lastCor.size() >= histlen)
                full = true;
        if (!full)
            return 0;

        p = (histlen - lastSum) / histlen;
        s = Math.sqrt(p * (1 - p) / histlen);
        if (p + s < pmin + smin) {
            pmin = p;
            smin = s;
            warning = 0;
        } else {
            if (p + s > pmin + warnLevel * smin) {
                warning++;
            } else
                warning = 0;
        }
        if (p + s > pmin + signLevel * smin)
            return -warning;
        else
            return warning;

    }

    public Iterator<ClassifyData> iterator() {
        return lastCor.iterator();
    }

}
