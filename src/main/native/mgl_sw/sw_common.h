#ifndef SW_COMMON_H
#define SW_COMMON_H


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string>
#include <list>

#if defined(_MSC_VER)
#include <intrin.h> // SIMD intrinsics for Windows
#else
#include <x86intrin.h> // SIMD intrinsics for GCC
#endif


//#define DEBUG_PRINT

#define SW_OS_SOFTCLIP  0x01 
#define SW_OS_INDEL     0x02 
#define SW_OS_LEAD_ID   0x04
#define SW_OS_IGNORE    0x08 

#define STATE_MATCH 'M'
#define STATE_INS 'I'
#define STATE_DEL 'D'
#define STATE_CLIP 'S'


#define SW_NEG_INF -0x40000000


struct ScoreMax{
	int mqe = SW_NEG_INF, mqe_t = -1;					// max score in last column and it's position
	int max = SW_NEG_INF, max_t = -1, max_q = -1;       // max score in last column and last row, and it's poition. If two scores are equal, the one closer to diagonal gets picked
	int seg_length = 0;				// calculated when we choose max in last row and last column
};

struct swParameters {
	int sc_match;
	int sc_mismatch;
	int g_open;
	int g_ext;
};

struct CigarElement
{
	char state;
	int length;
	CigarElement(char a, int b) {
		length = b; state = a; ;
	}
};

static inline void resetScoreMax(ScoreMax *ez)
{
	ez->mqe_t = ez->max_q = ez->max_t = -1;
	ez->mqe = ez->max = SW_NEG_INF;
};

#ifdef DEBUG_PRINT
static void print_array(FILE* f, int * a, int n)
{
	for (int i = 0; i < n; i++)
		fprintf(f, "%d\t", a[i]);
	fprintf(f, "\n");
};
#endif

#endif //SW_COMMON_H


