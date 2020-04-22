package com.microsoft.mgl.smithwaterman;

import com.intel.gkl.smithwaterman.IntelSmithWaterman;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.broadinstitute.gatk.nativebindings.smithwaterman.SWAlignerNativeBinding;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWNativeAlignerResult;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWOverhangStrategy;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWParameters;

import java.io.Console;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

public class CompareSmithWaterman implements SWAlignerNativeBinding {
    private SWAlignerNativeBinding binding1;
    private SWAlignerNativeBinding binding2;

    public CompareSmithWaterman() {
        binding1 = new IntelSmithWaterman();
        binding2 = new MicrosoftSmithWaterman();
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
        boolean isLoaded1 = binding1.load(tempDir);
        boolean isLoaded2 = binding2.load(tempDir);

        return (isLoaded1 && isLoaded2);
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
        SWNativeAlignerResult sw1 = binding1.align(refArray, altArray, parameters, overhangStrategy);

        SWNativeAlignerResult sw2 = binding2.align(refArray, altArray, parameters, overhangStrategy);

        if(sw1.alignment_offset != sw2.alignment_offset) {
            System.err.println("Mismatch: " + sw1.alignment_offset + "\t\t" + sw2.alignment_offset);
        }

        /*
        for(int i=0; i<likelihoodArray.length; i++)
        {
            if(Math.abs(likelihoodArray[i]-likelihoodArray2[i]) > 1.e-5 ) {
                System.err.println("Mismatch: " + likelihoodArray[i] + "\t\t" + likelihoodArray2[i]);
            }
        }
        */

        return sw1;
    }

    /**
     *
     */
    @Override
    public void close() {
        binding1.close();
        binding2.close();
    }
}
