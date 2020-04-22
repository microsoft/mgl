/*Copyright (c) 2012 The Broad Institute

*Permission is hereby granted, free of charge, to any person
*obtaining a copy of this software and associated documentation
*files (the "Software"), to deal in the Software without
*restriction, including without limitation the rights to use,
*copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the
*Software is furnished to do so, subject to the following
*conditions:

*The above copyright notice and this permission notice shall be
*included in all copies or substantial portions of the Software.

*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
*EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
*OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
*NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
*HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
*WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
*THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <regex>
#include <vector>

#include "cpuid.h"
#include "pairhmm_common.h"
#include "com_microsoft_mgl_pairhmm_MicrosoftPairHmm.h"
#include "compute_prob_scalar.h"
#include "compute_prob_avx.h"

using namespace std;

bool g_use_double;
int g_max_threads;
bool g_use_fpga;

Context<float> g_ctxf;
Context<double> g_ctxd;

void(*g_compute_prob_float)(readinfo &read, vector<hapinfo> &hap_array) = compute_prob_scalarf;
void(*g_compute_prob_double)(readinfo &read, vector<hapinfo> &hap_array) = compute_prob_scalard;

/*
* Class:     com_microsoft_mgl_pairhmm_MicrosoftPairHmm
* Method:    initNative
* Signature: (ZI)V
*/
JNIEXPORT void JNICALL Java_com_microsoft_mgl_pairhmm_MicrosoftPairHmm_initNative
(JNIEnv* env, jclass cls,
	jboolean use_double, jint max_threads)
{
	g_use_double = use_double;

	// enable FTZ
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

	//// set function pointers
	//if (is_avx512_supported())
	//{
	//	g_compute_prob_float = compute_fp_avx512s;
	//	g_compute_prob_double = compute_fp_avx512d;
	//}
	if (cpuid::has_AVX2)
	{
		g_compute_prob_float = compute_prob_avxf;
		g_compute_prob_double = compute_prob_avxd;
	}
}

/*
* Class:     com_microsoft_mgl_pairhmm_MicrosoftPairHmm
* Method:    computeLikelihoodsNative
* Signature: (Ljava/nio/IntBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/DoubleBuffer;)V
*/
JNIEXPORT void JNICALL Java_com_microsoft_mgl_pairhmm_MicrosoftPairHmm_computeLikelihoodsNative
(JNIEnv* env, jobject obj,
	jobject lengthBuffer, jobject readsBuffer, jobject haplotypesBuffer, jobject likelihoodBuffer)
{
	//==================================================================
	// get Java data
	jint *pLengths = (jint *)env->GetDirectBufferAddress(lengthBuffer);

	jint readCount = *pLengths++;
	vector<readinfo> reads(readCount);

#ifdef _DEBUG
	printf("%d ", readCount);
#endif
	char* pReads = (char *)env->GetDirectBufferAddress(readsBuffer);

	for (jint i = 0; i < readCount; i++)
	{
		jint readLen = *pLengths++;
		reads[i].rslen = readLen;
		reads[i].rs = pReads; pReads += readLen;
		reads[i].q = pReads; pReads += readLen;
		reads[i].i = pReads; pReads += readLen;
		reads[i].d = pReads; pReads += readLen;
		reads[i].c = pReads; pReads += readLen;
	}

	jint hapCount = *pLengths++;
	vector<hapinfo> haplotypes(hapCount);

#ifdef _DEBUG
	printf("%d\n", hapCount);
#endif
	char* pHaplotypes = (char *)env->GetDirectBufferAddress(haplotypesBuffer);

	for (jint j = 0; j < hapCount; j++)
	{
		jint hapLen = *pLengths++;
		haplotypes[j].haplen = hapLen;
		haplotypes[j].hap = pHaplotypes; pHaplotypes += hapLen;
		haplotypes[j].index = j;
		haplotypes[j].position = 0;
		haplotypes[j].score = 0.0;
	}

	jdouble* likelihoodArray = (jdouble *)env->GetDirectBufferAddress(likelihoodBuffer);

	//==================================================================
	// calcutate pairHMM

//#ifdef _DEBUG
//	tbb::blocked_range<jint> r(0, readCount);
//#else
//#endif
	tbb::parallel_for(tbb::blocked_range<jint>(0, readCount),
		[&](const tbb::blocked_range<jint>& r) {
		for (jint read_idx = r.begin(); read_idx != r.end(); read_idx++)
		{
			size_t numNonMatching = 0;

			vector<hapinfo> hap_nonmatching;
			hap_nonmatching.resize(hapCount);

			//if (g_compute_fast_prob_float != NULL)
			//{
			//    g_compute_fast_prob_float(reads[read_idx]), haplotypes);
			//}

			for (size_t hap_idx = 0; hap_idx < hapCount; ++hap_idx)
			{
				double result_fast = haplotypes[hap_idx].score;

				if (result_fast < MIN_ACCEPTED)
				{
					hap_nonmatching[numNonMatching++] = haplotypes[hap_idx];
				}
				else
				{
					double result = log10(result_fast) - g_ctxf.LOG10_INITIAL_CONSTANT;

					likelihoodArray[read_idx * hapCount + hap_idx] = result;
				}
			}

			hap_nonmatching.resize(numNonMatching);

			vector<hapinfo> hap_double;
			hap_double.resize(numNonMatching);

			size_t numDouble = 0;

			if (numNonMatching > 0)
			{
				if (g_use_double)
				{
					for (size_t hap_idx = 0; hap_idx < numNonMatching; ++hap_idx)
					{
						hap_double[numDouble++] = hap_nonmatching[hap_idx];
					}
				}
				else
				{
					g_compute_prob_float(reads[read_idx], hap_nonmatching);

					for (size_t hap_idx = 0; hap_idx < numNonMatching; ++hap_idx)
					{
						double result_avxf = hap_nonmatching[hap_idx].score;

						if (result_avxf < MIN_ACCEPTED)
						{
							hap_double[numDouble++] = hap_nonmatching[hap_idx];
						}
						else
						{
							double result = log10(result_avxf) - g_ctxf.LOG10_INITIAL_CONSTANT;

							likelihoodArray[read_idx * hapCount + hap_nonmatching[hap_idx].index] = result;
						}
					}
				}
			}

			if (numDouble > 0)
			{
				hap_double.resize(numDouble);

				g_compute_prob_double(reads[read_idx], hap_double);

				for (size_t hap_idx = 0; hap_idx < numDouble; ++hap_idx)
				{
					double result_avxd = hap_double[hap_idx].score;

					double result = log10(result_avxd) - g_ctxd.LOG10_INITIAL_CONSTANT;

					likelihoodArray[read_idx * hapCount + hap_double[hap_idx].index] = result;
				}
			}
		}
	});
//#ifndef _DEBUG
//#endif

#ifdef _DEBUG
	printf("Done \n");
#endif
}

/*
* Class:     com_microsoft_mgl_pairhmm_MicrosoftPairHmm
* Method:    doneNative
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_com_microsoft_mgl_pairhmm_MicrosoftPairHmm_doneNative
(JNIEnv* env, jobject obj)
{
#ifdef _DEBUG
	printf("Done Native\n");
#endif
}
