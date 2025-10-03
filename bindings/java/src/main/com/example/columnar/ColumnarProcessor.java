package com.example.columnar;

public class ColumnarProcessor {
    private long nativeHandle;

    public ColumnarProcessor() {
        nativeHandle = nativeNew();
    }

    private static native long nativeNew();
    public native int nativeLoadCsv(String path);
    public native int nativeRowCount();

    // Cleanup
    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        // optionally free Rust object
    }

    static {
        System.loadLibrary("columnar_processor"); // Rust shared lib
    }
}
