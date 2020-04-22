package com.microsoft.mgl.pairhmm;


import com.intel.gkl.IntelGKLUtils;
import com.intel.gkl.NativeLibraryLoader;
import com.intel.gkl.pairhmm.IntelPairHmm;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.gatk.nativebindings.pairhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeBinding;
import org.broadinstitute.gatk.nativebindings.pairhmm.ReadDataHolder;

import java.io.Console;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

public class ComparePairHmm implements PairHMMNativeBinding {
    private PairHMMNativeBinding binding1;
    private PairHMMNativeBinding binding2;

    public ComparePairHmm() {
        binding1 = new IntelPairHmm();
        binding2 = new MicrosoftPairHmm();
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
     * Initialize native PairHMM with the supplied args.
     *
     * @param args the args used to configure native PairHMM
     */
    @Override
    public void initialize(PairHMMNativeArguments args) {
        binding1.initialize(args);
        binding2.initialize(args);
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
        binding1.computeLikelihoods(readDataArray, haplotypeDataArray, likelihoodArray);

        double[] likelihoodArray2 = new double[likelihoodArray.length];

        binding2.computeLikelihoods(readDataArray, haplotypeDataArray, likelihoodArray2);

        for(int i=0; i<likelihoodArray.length; i++)
        {
            if(Math.abs(likelihoodArray[i]-likelihoodArray2[i]) > 1.e-5 ) {
                 System.err.println("Mismatch: " + likelihoodArray[i] + "\t\t" + likelihoodArray2[i]);
            }
        }
    }

    /**
     *
     */
    @Override
    public void done() {
        binding1.done();
        binding2.done();
    }
}
