package utils.stats;

import java.util.Comparator;
import java.util.Iterator;
import java.util.TreeSet;

public class ClassifyTTest {

    protected double pmin = Double.MAX_VALUE;

    protected double smin = Double.MAX_VALUE;

    protected int histlen = 100;

    protected double lastSum = 0;

    protected int lastPos = 0;

    protected int classifyCount = 0;

    protected int lastHist = 20;

    protected int warning;

    protected float[] lastCor;

    protected double p, s;

    protected boolean full;

    protected double warnLevel;

    protected double signLevel;

    public ClassifyTTest(double warnLevel, double signLevel, int histlen, int lastHist) {
        this.warnLevel = warnLevel;
        this.signLevel = signLevel;
        this.histlen = histlen;
        this.lastHist = lastHist;
        init();
    }

    public ClassifyTTest(ClassifyTTest test) {
        this.warnLevel = test.warnLevel;
        this.signLevel = test.signLevel;
        this.histlen = test.histlen;
        init();
        System.arraycopy(test.lastCor, 0, lastCor, 0, histlen);
        this.p = test.p;
        this.s = test.s;
        this.pmin = test.pmin;
        this.smin = test.smin;
        this.lastSum = test.lastSum;
        this.lastPos = test.lastPos;
        this.full=test.full;
        this.classifyCount=test.classifyCount;
    }

    public void init() {
        full = false;
        lastCor = new float[histlen];
        lastSum = 0;
        pmin = Double.MAX_VALUE;
        smin = Double.MAX_VALUE;
        warning = 0;
        lastPos = 0;
        classifyCount = 0;
    }

    public boolean isSignificant() {
        return full && p + s > pmin + signLevel * smin && warning>=lastHist-2;
    }

    public double[] getCurrentValues() {
        return new double[] { p, s };
    }
    
    public double getCurrentError(){
        return p;
    }

    public double[] getMinValues() {
        return new double[] { pmin, smin };
    }

    public int getWarning() {
        return warning;
    }

    public void resetStats(double p) {
        pmin = Double.MAX_VALUE;
        smin = Double.MAX_VALUE;
        classifyCount = histlen-lastHist;
        full=false;
        // lastCor.clear();
        warning = 0;
    }

    public void addOptimize(double fak) {
        pmin = pmin + fak;
        smin = Math.sqrt(pmin * (1 - pmin) / histlen);
    }

    public double getProbDiff() {
        int len = lastHist;
        int newLen = Math.max(0, len);
        // if (warning>histlen-len){
        // return p-pmin;
        // }
        int sumNew = 0;
        for (int i = 0; i < newLen; i++) {
            sumNew += lastCor[(lastPos-1 - i + histlen) % histlen];
        }
        return (1.0 - pmin) - (sumNew / (double) newLen);
    }
    
    public double getLastError(int window) {
        if (window>histlen) return Double.NaN;
        if (window>classifyCount && !full) return 0;
        int sumNew = 0;
        
        for (int i = 0; i < window; i++) {
            sumNew += lastCor[(lastPos-1 - i + histlen) % histlen];
        }
        return ((window-sumNew) / (double) window);
    }

    public int addClassify(float correct) {
        lastSum -= lastCor[lastPos];
        lastCor[lastPos] = correct;
        lastSum += correct;
        if (!full && classifyCount>lastPos){
            System.out.println();
        }
        lastPos = (lastPos + 1) % histlen;
        if (classifyCount < histlen) {
            classifyCount++;
            if (classifyCount >= histlen)
                full = true;
        }
        if (!full && classifyCount>lastPos){
            System.out.println();
        }
        if (!full)
            return 0;

        p = (histlen - lastSum) / histlen;
        s = Math.sqrt(p * (1 - p) / classifyCount);
        if (p + s < pmin + smin) {
            pmin = p;
            smin = s;
            warning = 0;
        } else {
            if (p + s > pmin + 2 * smin) {
                warning++;
            } else
                warning = 0;
        }

        if (p + s > pmin + signLevel * smin)
            return -warning;
        else
            return warning;

    }

   
}
