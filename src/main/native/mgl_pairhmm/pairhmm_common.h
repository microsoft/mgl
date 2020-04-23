#ifndef PAIRHMM_COMMON_H
#define PAIRHMM_COMMON_H

#if defined(_MSC_VER)
  #include <intrin.h> // SIMD intrinsics for Windows
#else
  #include <x86intrin.h> // SIMD intrinsics for GCC
#endif

#include <vector>
#include <cassert>
#include <cstdint>

#include "Context.h"

#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>

#ifdef _DEBUG
#define _UNIT_TEST
//#define _UNIT_TEST_DUMP
#endif

#define MM 0
#define GapM 1
#define MX 2
#define XX 3
#define MY 4
#define YY 5

#define MIN_ACCEPTED 1e-28f
#define NUM_DISTINCT_CHARS 5
#define AMBIG_CHAR 4

//typedef struct {
//  int rslen, haplen;
//  const char *q, *i, *d, *c;
//  const char *hap, *rs;
//} testcase;
//
//
struct hapinfo
{
	size_t haplen;
	char *hap;
	size_t index;
	size_t position;
	double score;
};

struct readinfo
{
	int rslen;
	char *q, *i, *d, *c;
	char *rs;
	int *irs;
};

void computeHaplotypeSimilarities(std::vector<hapinfo> & hap_array, size_t &COLS_MIN, size_t &COLS_MAX);

#ifdef _UNIT_TEST
template<class NUMBER>
void DebugAssertClose(NUMBER a, NUMBER b)
{
	NUMBER delta = NUMBER(1.0e-5);

	assert(((a < NUMBER(1.0)) && (b < NUMBER(1.0))) || ((a == NUMBER(0.0)) && (abs(b) < delta)) || ((b == NUMBER(0.0)) && (abs(a) < delta)) || (abs((b - a) / a) < delta));
}

template<class NUMBER>
void DebugVerify(
	size_t startRow, size_t startCol, size_t iterations,
	NUMBER * MDiag0, NUMBER * XDiag0, NUMBER * YDiag0,
	NUMBER ** M, NUMBER ** X, NUMBER ** Y)
{
	for (size_t r = startRow + iterations, c = startCol - iterations; iterations > 0; iterations--, r--, c++)
	{
		DebugAssertClose(MDiag0[r], M[r][c]);
		DebugAssertClose(XDiag0[r], X[r][c]);
		DebugAssertClose(YDiag0[r], Y[r][c]);
	}
}

template<class NUMBER>
NUMBER** DebugCompute(Context<NUMBER> &ctx,
	readinfo &read, hapinfo &haplotype,
	NUMBER yInitial)
{
	size_t ROWS = read.rslen + 1;
	size_t COLS = haplotype.haplen + 1;

	//allocate on heap in way that simulates a 2D array. Having a 2D array instead of
	//a straightforward array of pointers ensures that all data lies 'close' in memory, increasing
	//the chance of being stored together in the cache. Also, prefetchers can learn memory access
	//patterns for 2D arrays, not possible for array of pointers
	//NUMBER* common_buffer = 0;
	NUMBER* common_buffer = new NUMBER[3 * ROWS*COLS + ROWS * 6];
	//pointers to within the allocated buffer
	NUMBER** common_pointer_buffer = new NUMBER*[4 * ROWS];
	NUMBER* ptr = common_buffer;

	int i = 0;
	for (; i < 3 * ROWS; ++i, ptr += COLS)
		common_pointer_buffer[i] = ptr;
	for (; i < 4 * ROWS; ++i, ptr += 6)
		common_pointer_buffer[i] = ptr;

	NUMBER** M = common_pointer_buffer;
	NUMBER** X = M + ROWS;
	NUMBER** Y = X + ROWS;
	NUMBER** p = Y + ROWS;

	p[0][MM] = ctx._(0.0);
	p[0][GapM] = ctx._(0.0);
	p[0][MX] = ctx._(0.0);
	p[0][XX] = ctx._(0.0);
	p[0][MY] = ctx._(0.0);
	p[0][YY] = ctx._(0.0);

	for (size_t r = 1; r < ROWS; r++)
	{
		int _i = read.i[r - 1] & 127;
		int _d = read.d[r - 1] & 127;
		int _c = read.c[r - 1] & 127;
		p[r][MM] = ctx.set_mm_prob(_i, _d);
		p[r][GapM] = ctx._(1.0) - ctx.ph2pr[_c];
		p[r][MX] = ctx.ph2pr[_i];
		p[r][XX] = ctx.ph2pr[_c];
		p[r][MY] = ctx.ph2pr[_d];
		p[r][YY] = ctx.ph2pr[_c];
	}

	for (size_t c = 0; c < COLS; c++)
	{
		M[0][c] = ctx._(0.0);
		X[0][c] = ctx._(0.0);
		Y[0][c] = yInitial;
	}

	for (size_t r = 1; r < ROWS; r++)
	{
		M[r][0] = ctx._(0.0);
		X[r][0] = X[r - 1][0] * p[r][XX];
		Y[r][0] = ctx._(0.0);
	}

	const NUMBER threeOver = ctx._(1.0) / ctx._(3.0);

	for (size_t r = 1; r < ROWS; r++)
	{
		for (size_t c = 1; c < COLS; c++)
		{
			char _rs = read.rs[r - 1];
			char _hap = haplotype.hap[c - 1];
			int _q = read.q[r - 1] & 127;
			NUMBER distm = ctx.ph2pr[_q];
			if (_rs == _hap || _rs == 'N' || _hap == 'N')
				distm = ctx._(1.0) - distm;
			else
				distm = distm * threeOver;

			M[r][c] = distm * (M[r - 1][c - 1] * p[r][MM] + (X[r - 1][c - 1] + Y[r - 1][c - 1]) * p[r][GapM]);

			X[r][c] = M[r - 1][c] * p[r][MX] + X[r - 1][c] * p[r][XX];

			Y[r][c] = M[r][c - 1] * p[r][MY] + Y[r][c - 1] * p[r][YY];
		}
	}

	return common_pointer_buffer;
}

void DebugDump(
	float** M, float ** X, float ** Y,
	size_t ROWS, size_t COLS);

void DebugDump(
	double** M, double ** X, double ** Y,
	size_t ROWS, size_t COLS);

#endif // _UNIT_TEST

#endif // PAIRHMM_COMMON_H
