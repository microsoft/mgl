package com.microsoft.mgl.pairhmm;


import com.microsoft.mgl.NativeLibraryLoader;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.gatk.nativebindings.pairhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeBinding;
import org.broadinstitute.gatk.nativebindings.pairhmm.ReadDataHolder;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

public class MicrosoftPairHmm implements PairHMMNativeBinding {
    private final static Logger logger = LogManager.getLogger(MicrosoftPairHmm.class);
    private static final String NATIVE_LIBRARY_NAME = "mgl_pairhmm;tbb;tbbmalloc";

    public MicrosoftPairHmm() {
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

    /**
     * Initialize native PairHMM with the supplied args.
     *
     * @param args the args used to configure native PairHMM
     */
    @Override
    public void initialize(PairHMMNativeArguments args) {
        if (args == null) {
            args = new PairHMMNativeArguments();
            args.useDoublePrecision = false;
            args.maxNumberOfThreads = 1;
        }

        initNative(args.useDoublePrecision, args.maxNumberOfThreads);
    }

    /**
     *
     *
     * @param readDataArray array of read data
     * @param haplotypeDataArray array of haplotype data
     * @param likelihoodArray array of double results
     */
    @Override
    public void computeLikelihoods(ReadDataHolder[] readDataArray,
                                   HaplotypeDataHolder[] haplotypeDataArray,
                                   double[] likelihoodArray)
    {

        int nReads = readDataArray.length;
        int nHaplotypes = haplotypeDataArray.length;
        ByteBuffer lengthByteBuffer = ByteBuffer.allocateDirect((nReads + nHaplotypes + 2) * 4);
        lengthByteBuffer.order(ByteOrder.nativeOrder());

        IntBuffer lengthBuffer = lengthByteBuffer.asIntBuffer();

        int nReadsLength =0;
        lengthBuffer.put(nReads);
        for(int i=0; i<nReads; i++) {
            int nReadLength = readDataArray[i].readBases.length;
            lengthBuffer.put(nReadLength);
            nReadsLength += nReadLength;
        }

        int nHaplotypesLength = 0;
        lengthBuffer.put(nHaplotypes);
        for(int j=0; j<nHaplotypes; j++) {
            int nHaplotypeLength = haplotypeDataArray[j].haplotypeBases.length;
            lengthBuffer.put(nHaplotypeLength);
            nHaplotypesLength += nHaplotypeLength;
        }

        ByteBuffer readsBuffer = ByteBuffer.allocateDirect(nReadsLength*5);
        for(int i=0; i<nReads; i++) {
            readsBuffer.put(readDataArray[i].readBases);
            readsBuffer.put(readDataArray[i].readQuals);
            readsBuffer.put(readDataArray[i].insertionGOP);
            readsBuffer.put(readDataArray[i].deletionGOP);
            readsBuffer.put(readDataArray[i].overallGCP);
        }

        ByteBuffer haplotypesBuffer = ByteBuffer.allocateDirect(nHaplotypesLength);
        for(int j=0; j<nHaplotypes; j++) {
            haplotypesBuffer.put(haplotypeDataArray[j].haplotypeBases);
        }

        ByteBuffer likelihoodByteBuffer = ByteBuffer.allocateDirect(likelihoodArray.length * 8);
        likelihoodByteBuffer.order(ByteOrder.nativeOrder());

        DoubleBuffer likelihoodBuffer = likelihoodByteBuffer.asDoubleBuffer();

        computeLikelihoodsNative(lengthBuffer, readsBuffer, haplotypesBuffer, likelihoodBuffer);

        likelihoodBuffer.get(likelihoodArray);
    }

    /**
     *
     */
    @Override
    public void done() {
        doneNative();
    }

    private native static void initNative(boolean doublePrecision,
                                          int maxThreads);

    private native void computeLikelihoodsNative(IntBuffer lengthBuffer,
                                                 ByteBuffer readsBuffer,
                                                 ByteBuffer haplotypesBuffer,
                                                 DoubleBuffer likelihoodBuffer);

    private native void doneNative();
}
