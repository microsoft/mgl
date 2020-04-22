/*++

Module Name:

    com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman.cpp

Abstract:

    GATK JNI Smith-Waterman AVX2 implementation functions.

Authors:

    Roman Snytsar, November, 2018

Environment:
`
    User mode service.

Revision History:


--*/
#include <regex>
#include "cpuid.h"
#include "com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman.h"
#include "sw_common.h"
#include "sw_scalar.h"
#include "sw_avx.h"

using namespace std;

/*
 * Class:     com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman
 * Method:    initNative
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman_initNative
(JNIEnv * env, jclass cls) {}

/*
 * Class:     com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman
 * Method:    alignNative
 * Signature: (Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIIIIII)I
 */
JNIEXPORT jint JNICALL Java_com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman_alignNative
(JNIEnv * env, jclass cls, jobject readsBuffer, jobject cigarBuffer, jint targetLength, jint queryLength, jint match, jint mismatch, jint gapOpen, jint gapExtend, jint overhangStrategy)
{

    char *target = (char *)env->GetDirectBufferAddress(readsBuffer), *query;
    query = target + targetLength;

    swParameters parameters;
    parameters.sc_match = match > 0? match : -match;
    parameters.sc_mismatch = mismatch < 0 ? mismatch : -mismatch;
    parameters.g_open = gapOpen > 0 ? gapOpen : -gapOpen;
    parameters.g_ext = gapExtend > 0 ? gapExtend : -gapExtend;

   string cigar;

    jint offset = 0;
// we skip substring check since it performed by SWNativeAlignerWrapper
 
   	if ((cpuid::has_AVX2) && (queryLength >= 8))
   	{
   		offset =  align_avx(target, targetLength, query, queryLength, parameters, overhangStrategy, &cigar);
		cigar.copy((char*)env->GetDirectBufferAddress(cigarBuffer), cigar.length());
		return offset;
   	}
   	offset =  align_scalar(target, targetLength, query, queryLength, parameters, overhangStrategy, &cigar);
	cigar.copy((char*)env->GetDirectBufferAddress(cigarBuffer), cigar.length());
   	return offset;
 }

/*
 * Class:     com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman
 * Method:    doneNative
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman_doneNative
(JNIEnv * env, jclass cls) {}
