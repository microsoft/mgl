package com.microsoft.mgl.smithwaterman;


import com.microsoft.mgl.NativeLibraryLoader;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.broadinstitute.gatk.nativebindings.smithwaterman.SWAlignerNativeBinding;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWNativeAlignerResult;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWOverhangStrategy;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWParameters;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

public class MicrosoftSmithWaterman implements SWAlignerNativeBinding {
    private final static Logger logger = LogManager.getLogger(MicrosoftSmithWaterman.class);
    private static final String NATIVE_LIBRARY_NAME = "mgl_sw";

    public MicrosoftSmithWaterman() {
    }

    /**
     * Loads the native library, if it is supported on this platform. <p>
     * Returns false if AVX is not supported. <br>
     * Returns false if the native library cannot be loaded for any reason. <br>
     *
     * @param tempDir  directory where the native library is extracted or null to use the system temp directory
     * @return  true if the native library is supported and loaded, false otherwise
     */
    @Override
    public synchronized boolean load(File tempDir) {
        return NativeLibraryLoader.load(tempDir == null ? null : tempDir.toPath(), NATIVE_LIBRARY_NAME);
    }

    private int getStrategy(SWOverhangStrategy strategy)
    {
        int intStrategy = 0;

        switch(strategy){
            case SOFTCLIP: intStrategy = 0x01;
                break;
            case INDEL: intStrategy = 0x02;
                break;
            case LEADING_INDEL: intStrategy = 0x04;
                break;
            case IGNORE: intStrategy = 0x08;
                break;
        }

        return intStrategy;

    }

    /**
     *Implements the native implementation of SmithWaterman, and returns the Cigar String and alignment_offset
     *
     * @param refArray array of reference data
     * @param altArray array of alternate data
     *
     */
    @Override
    public SWNativeAlignerResult align(byte[] refArray, byte[] altArray, SWParameters parameters, SWOverhangStrategy overhangStrategy)
    {
        int refLength = refArray.length;
        int altLength = altArray.length;

        byte[] cigar = new byte[2*Integer.max(refLength, altLength)];

        ByteBuffer readsBuffer = ByteBuffer.allocateDirect(refLength + altLength);
        readsBuffer.put(refArray);
        readsBuffer.put(altArray);

        ByteBuffer cigarBuffer = ByteBuffer.allocateDirect(cigar.length);

        int intStrategy =  getStrategy(overhangStrategy);

        int offset = alignNative(readsBuffer, cigarBuffer, refLength, altLength, parameters.getMatchValue(), parameters.getMismatchPenalty(), parameters.getGapOpenPenalty(), parameters.getGapExtendPenalty(), intStrategy);

        cigarBuffer.get(cigar);

        return new SWNativeAlignerResult(new String(cigar).trim(), offset);
    }

    @Override
    public void close()
    {
        doneNative();
    }

    private native static void initNative();
    private native static int alignNative(ByteBuffer readsBuffer, ByteBuffer cigarBuffer, int refLength, int altLength, int match, int mismatch, int open, int extend, int strategy);
    private native static void doneNative();
}
