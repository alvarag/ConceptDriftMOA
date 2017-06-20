package utils.indexstructure;

import java.io.Serializable;

public interface DoubleFunction<T> extends Serializable {
    public double invoke(T t);
}