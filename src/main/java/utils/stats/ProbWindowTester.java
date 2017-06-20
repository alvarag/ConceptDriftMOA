package utils.stats;

public class ProbWindowTester {
    
    int winsize1;
    int winsize2;
    double warning;
    double error;
    
    double pmin;
    double smin;
    double p;
    double s;
    
    boolean[] last;
    int pos;
    int correct1;
    int correct2;
    int state;

    public ProbWindowTester(int winsize1, int winsize2, double warning, double error){
        this.error=error;
        this.winsize1=winsize1;
        this.winsize2=winsize2;
        this.warning=warning;
        reset();
    }
    
    public void reset(int winsize1, int winsize2, double warning, double error){
        this.error=error;
        this.winsize1=winsize1;
        this.winsize2=winsize2;
        this.warning=warning;
        reset();
    }
    
    public void reset(){
        last=new boolean[winsize1];
        pos=0;
        correct1=0;
        correct2=0;
        state=0;
    }
    
    public void add(boolean result){
        if (last[winsize2-1]) correct2--;
        if (last[winsize1-1]) correct1--;
        System.arraycopy(last,0,last,1,winsize1-1);
        last[0]=result;
        if (result){
            correct1++;
            correct2++;
        }
        if (pos<winsize1) pos++;
        
        p=1-(correct1/(double)pos);
        s=Math.sqrt(p*(1-p)/pos);
        if (pos<winsize2){
            state=0;
            return;
        }
        if (p<pmin){
            pmin=p;
            smin=s;
        }
        p=1-(correct2/(double)winsize2);
        s=Math.sqrt(p*(1-p)/winsize2);
        if(p+s>pmin+error*smin) 
            state=2;
        else if(p+s>pmin+warning*smin) state=1;
        else state=0;
    }
    
    public boolean isError(){
        return state==2;
    }
    
    public boolean isWarning(){
        return state>=1;
    }
    
}
