package com.microsoft.mgl.pairhmm;

import com.intel.gkl.IntelGKLUtils;
import com.intel.gkl.NativeLibraryLoader;
import com.microsoft.mgl.pairhmm.MicrosoftPairHmm;
import org.broadinstitute.gatk.nativebindings.pairhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeBinding;
import org.broadinstitute.gatk.nativebindings.pairhmm.ReadDataHolder;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.*;
import java.util.Scanner;

public class ComparePairHmmUnitTest {
    static final String pairHMMTestData = IntelGKLUtils.pathToTestResource("pairhmm-testdata.txt");

    @Test(enabled = true)
    public void simpleTest() {

        final boolean isloaded = new ComparePairHmm().load(null);

        final ComparePairHmm pairHmm = new ComparePairHmm();
        Assert.assertTrue(isloaded);

        final PairHMMNativeArguments args = new PairHMMNativeArguments();
        args.maxNumberOfThreads = 1;
        args.useDoublePrecision = false;

        pairHmm.initialize(args);

        ReadDataHolder[] readDataArray = new ReadDataHolder[1];
        HaplotypeDataHolder[] haplotypeDataArray = new HaplotypeDataHolder[1];
        double[] likelihoodArray = new double[1];

        // initialize data in place
        haplotypeDataArray[0] = new HaplotypeDataHolder();
        haplotypeDataArray[0].haplotypeBases = "ACGT".getBytes();
        readDataArray[0] = new ReadDataHolder();
        readDataArray[0].readBases = "ACGT".getBytes();
        readDataArray[0].readQuals = "++++".getBytes();
        readDataArray[0].insertionGOP = "++++".getBytes();
        readDataArray[0].deletionGOP = "++++".getBytes();
        readDataArray[0].overallGCP = "++++".getBytes();
        double expectedResult = -6.022797e-01;

        // call pairHMM
        pairHmm.computeLikelihoods(readDataArray, haplotypeDataArray, likelihoodArray);

        // check result
        Assert.assertEquals(likelihoodArray[0], expectedResult, 1e-5, "Likelihood not in expected range.");
    }

    @Test(enabled = true)
    public void dataFileTest() {
        // load native library
        final boolean isloaded = new ComparePairHmm().load(null);
        Assert.assertTrue(isloaded);

        boolean[] udvals = {false, true};
        for(boolean useDbl : udvals) {
            // instantiate and initialize ComparePairHmm
            final ComparePairHmm pairHmm = new ComparePairHmm();

            final PairHMMNativeArguments args = new PairHMMNativeArguments();
            args.maxNumberOfThreads = 1;
            args.useDoublePrecision = useDbl;
            pairHmm.initialize(args);

            // data structures
            ReadDataHolder[] readDataArray = new ReadDataHolder[1];
            readDataArray[0] = new ReadDataHolder();
            HaplotypeDataHolder[] haplotypeDataArray = new HaplotypeDataHolder[1];
            haplotypeDataArray[0] = new HaplotypeDataHolder();
            double[] likelihoodArray = new double[1];

            // read test data from file
            Scanner s = null;
            try {
                s = new Scanner(new BufferedReader(new FileReader(pairHMMTestData)));

                while (s.hasNext()) {
                    // skip comment lines
                    if(s.hasNext("#.*")) {
                        s.nextLine();
                        continue;
                    }

                    haplotypeDataArray[0].haplotypeBases = s.next().getBytes();
                    readDataArray[0].readBases = s.next().getBytes();
                    readDataArray[0].readQuals = normalize(s.next().getBytes(), 6);
                    readDataArray[0].insertionGOP = normalize(s.next().getBytes());
                    readDataArray[0].deletionGOP = normalize(s.next().getBytes());
                    readDataArray[0].overallGCP = normalize(s.next().getBytes());
                    double expectedResult = s.nextDouble();

                    // call pairHMM
                    pairHmm.computeLikelihoods(readDataArray, haplotypeDataArray, likelihoodArray);

                    // check result
                    Assert.assertEquals(likelihoodArray[0], expectedResult, 1e-5, "Likelihood not in expected range.");
                }
            } catch (FileNotFoundException e) {
                Assert.fail("File not found : " + pairHMMTestData);
            } catch (Exception e) {
                e.printStackTrace();
                Assert.fail("Unexpected exception");
            }

            s.close();
            pairHmm.done();
        }
    }

    static byte[] normalize(byte[] scores) {
        return normalize(scores, 0);
    }

    static byte[] normalize(byte[] scores, int min) {
        for (int i = 0; i < scores.length; i++) {
            scores[i] -= 33;
            scores[i] = scores[i] < min ? (byte)min : scores[i];
        }
        return scores;
    }
}
