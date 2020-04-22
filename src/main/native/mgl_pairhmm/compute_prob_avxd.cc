#include "compute_prob_scalar.h"
#include "compute_prob_avx.h"

using namespace std;

const size_t bandWidth_pd = 4;

template <size_t __N>
__forceinline void _mm256_shift_left_si256(__m256i& a, __m256i b) {
	__m256i c = _mm256_permute2x128_si256(a, b, 0x03);
	a = _mm256_alignr_epi8(a, c, 16 - __N);
}

template <size_t __N>
__forceinline void _mm256_shift_left_si256(__m256i& a) {
	__m256i c = _mm256_permute2x128_si256(a, _mm256_setzero_si256(), 0x03);
	a = _mm256_alignr_epi8(a, c, 16 - __N);
}

__forceinline __m256d _mm256_shift_left_pd(__m256d a, const __m256d b) {
	__m256i c = _mm256_permute2x128_si256(
		_mm256_castpd_si256(a),
		_mm256_castpd_si256(b),
		0x03);

	return _mm256_castsi256_pd(
		_mm256_alignr_epi8(
			_mm256_castpd_si256(a),
			c,
			8));
}

__forceinline void _mm256_shift_left_pd(__m256d& a) {
	a = _mm256_shift_left_pd(a, _mm256_setzero_pd());
}
__forceinline void ConvertChars_pd(const char* rs, __m256i &_rs)
{
	__m128i _rs128 = _mm_set1_epi64x((*((const __int32*)rs)) & 0x0f0f0f0fll);

	__m128i _rsConverted = _mm_shuffle_epi8(
		_mm_set_epi8(0, 4, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 0),
		_rs128);

	_rs = _mm256_cvtepi8_epi64(_rsConverted);
}

__forceinline __m256d ComputeDistm_pd(const __m256i _rs, const __m256d _distM, const __m256d _distN, __m256i _hap)
{
	__m256i _match = _mm256_or_si256(
		_mm256_cmpeq_epi64(_rs, _hap),
		_mm256_slli_epi64(
			_mm256_or_si256(_rs, _hap),
			61));

	return _mm256_blendv_pd(
		_distN,
		_distM,
		_mm256_castsi256_pd(_match)
	);
}

__forceinline __m256d ComputeDistm_pd(const __m256i _rs, const __m256d _distM, const __m256d _distN, int64_t* i_hap, size_t col)
{
	__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + col));

	return ComputeDistm_pd(_rs, _distM, _distN, _hap);
}

__forceinline void AdvanceCellFirstRow(__m256d &_M0, __m256d &_X0, __m256d &_Y0, __m256d &_M1, __m256d &_X1, __m256d &_Y1, __m256d &_YInitial, double * &pOutput, __m256d &Distm0, __m256d &_pMM, __m256d &_pMX, __m256d &_pMY, __m256d &_pZZ)
{
	_M0 = _mm256_mul_pd(
		Distm0,
		_mm256_fmadd_pd(
			_M0,
			_pMM,
			_mm256_add_pd(
				_X0,
				_Y0
			)));

	_mm256_store_pd(pOutput, _M0);

	_Y0 = _mm256_fmadd_pd(
		_M1,
		_pMY,
		_mm256_mul_pd(
			_Y1,
			_pZZ
		));

	_mm256_shift_left_pd(_M1);

	_mm256_shift_left_pd(_X1);

	_X0 = _mm256_fmadd_pd(
		_M1,
		_pMX,
		_mm256_mul_pd(
			_X1,
			_pZZ
		));

	_mm256_store_pd(pOutput + bandWidth_pd, _X0);

	_Y1 = _mm256_shift_left_pd(_Y1, _YInitial);

	_mm256_store_pd(pOutput + 2 * bandWidth_pd, _Y0);

	pOutput += 3 * bandWidth_pd;
}

__forceinline void AdvanceCell(__m256d &_M0, __m256d &_X0, __m256d &_Y0, __m256d &_M1, __m256d &_X1, __m256d &_Y1, double * &pInput, double * &pOutput, __m256d &Distm0, __m256d &_pMM, __m256d &_pMX, __m256d &_pMY, __m256d &_pZZ)
{
	_M0 = _mm256_mul_pd(
		Distm0,
		_mm256_fmadd_pd(
			_M0,
			_pMM,
			_mm256_add_pd(
				_X0,
				_Y0
			)));

	_mm256_store_pd(pOutput, _M0);

	_Y0 = _mm256_fmadd_pd(
		_M1,
		_pMY,
		_mm256_mul_pd(
			_Y1,
			_pZZ
		));

	_M1 = _mm256_shift_left_pd(_M1, _mm256_load_pd(pInput));

	_X1 = _mm256_shift_left_pd(_X1, _mm256_load_pd(pInput + bandWidth_pd));

	_X0 = _mm256_fmadd_pd(
		_M1,
		_pMX,
		_mm256_mul_pd(
			_X1,
			_pZZ
		));

	_mm256_store_pd(pOutput + bandWidth_pd, _X0);

	_Y1 = _mm256_shift_left_pd(_Y1, _mm256_load_pd(pInput + 2 * bandWidth_pd));

	_mm256_store_pd(pOutput + 2 * bandWidth_pd, _Y0);

	pInput += 3 * bandWidth_pd;
	pOutput += 3 * bandWidth_pd;
}

void compute_bulk_band_first(
	readinfo& read, hapinfo &hapinfo,
	const __m256i _rs, const __m256d _distM, const __m256d _distN,
	int64_t* i_hap,
	double* pMM,
	double* pRow0, double yInitial, double* pCache)
{
	int ROWS = read.rslen;

	double* pMX = pMM + ROWS;
	double* pMY = pMX + ROWS;
	double* pZZ = pMY + ROWS;

	__m256d _YInitial = _mm256_set1_pd(yInitial);

	size_t COLS = hapinfo.haplen;

	double* pOutput = pRow0;

	__m256d _pMM = _mm256_loadu_pd(pMM);

	__m256d _pMX = _mm256_loadu_pd(pMX);

	__m256d _pMY = _mm256_loadu_pd(pMY);

	__m256d _pZZ = _mm256_set1_pd(*pZZ);

	__m256d _M0, _X0, _Y0;
	__m256d _M1, _X1, _Y1;

	size_t i = 0;
	size_t i_r = COLS;

	if (hapinfo.position > 0)
	{
		size_t offset = 3 * bandWidth_pd * hapinfo.position;

		pOutput += offset;

		_M0 = _mm256_load_pd(pCache);
		_mm256_store_pd(pOutput - 6 * bandWidth_pd, _M0);
		_mm256_shift_left_pd(_M0);

		_X0 = _mm256_load_pd(pCache + bandWidth_pd);
		_mm256_store_pd(pOutput - 5 * bandWidth_pd, _X0);
		_mm256_shift_left_pd(_X0);

		_Y0 = _mm256_load_pd(pCache + 2 * bandWidth_pd);
		_mm256_store_pd(pOutput - 4 * bandWidth_pd, _Y0);
		_Y0 = _mm256_shift_left_pd(_Y0, _YInitial);


		_M1 = _mm256_load_pd(pCache + 3 * bandWidth_pd);
		_mm256_store_pd(pOutput - 3 * bandWidth_pd, _M1);

		_X1 = _mm256_load_pd(pCache + 4 * bandWidth_pd);
		_mm256_store_pd(pOutput - 2 * bandWidth_pd, _X1);

		_Y1 = _mm256_load_pd(pCache + 5 * bandWidth_pd);
		_mm256_store_pd(pOutput - bandWidth_pd, _Y1);

		i += hapinfo.position;
		i_r -= hapinfo.position - 1;
	}

	switch (hapinfo.position)
	{
	case 0:

		// -1
		_M1 = _mm256_setzero_pd();
		_X1 = _mm256_setzero_pd();
		_Y1 = _mm256_blend_pd(_YInitial, _mm256_setzero_pd(), 0x0E);

		// 0
		_M0 = _mm256_setzero_pd();
		_mm256_store_pd(pOutput, _M0);
		_X0 = _mm256_setzero_pd();
		_mm256_store_pd(pOutput + bandWidth_pd, _X0);
		_Y0 = _mm256_setzero_pd();
		_mm256_store_pd(pOutput + 2 * bandWidth_pd, _Y0);

		pOutput += 3 * bandWidth_pd;

		// 1
		__m256d Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	case 2:
		// 2
		Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		// 3
		Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	default:
	{
		for (; i < COLS; i += 2)
		{
			// Even
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// Odd
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}

		if (COLS % 2 == 0)
		{
			// Last Element - Even
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + i_r));

			// -2
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -1
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -0
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}
		else
		{
			__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + i_r));

			// -2
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -1
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -0
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}
	}
	}
}

void compute_bulk_band(
	readinfo &read, hapinfo &hapinfo,
	const __m256i _rs, const __m256d _distM, const __m256d _distN,
	int64_t* i_hap,
	double * pMM,
	double* pRow0, double* pRow1, double* pCache,
	size_t row)
{
	int ROWS = read.rslen;

	double* pMX = pMM + ROWS;
	double* pMY = pMX + ROWS;
	double* pZZ = pMY + ROWS;

	size_t COLS = hapinfo.haplen;

	// Skip forward triangle
	double* pInput = pRow0 + 3 * bandWidth_pd * (bandWidth_pd - 1);
	double* pOutput = pRow1;

	__m256d _pMM = _mm256_loadu_pd(pMM + row);

	__m256d _pMX = _mm256_loadu_pd(pMX + row);

	__m256d _pMY = _mm256_loadu_pd(pMY + row);

	__m256d _pZZ = _mm256_set1_pd(*pZZ);

	__m256d _M0, _X0, _Y0;
	__m256d _M1, _X1, _Y1;

	size_t i = 0;
	size_t i_r = COLS;

	if (hapinfo.position > 0)
	{
		size_t offset = 3 * bandWidth_pd * hapinfo.position;

		pOutput += offset;
		pInput += offset;

		_M0 = _mm256_load_pd(pCache);
		_mm256_store_pd(pOutput - 6 * bandWidth_pd, _M0);
		_M0 = _mm256_shift_left_pd(_M0, _mm256_load_pd(pInput - 3 * bandWidth_pd));

		_X0 = _mm256_load_pd(pCache + bandWidth_pd);
		_mm256_store_pd(pOutput - 5 * bandWidth_pd, _X0);
		_X0 = _mm256_shift_left_pd(_X0, _mm256_load_pd(pInput - 2 * bandWidth_pd));

		_Y0 = _mm256_load_pd(pCache + 2 * bandWidth_pd);
		_mm256_store_pd(pOutput - 4 * bandWidth_pd, _Y0);
		_Y0 = _mm256_shift_left_pd(_Y0, _mm256_load_pd(pInput - bandWidth_pd));


		_M1 = _mm256_load_pd(pCache + 3 * bandWidth_pd);
		_mm256_store_pd(pOutput - 3 * bandWidth_pd, _M1);

		_X1 = _mm256_load_pd(pCache + 4 * bandWidth_pd);
		_mm256_store_pd(pOutput - 2 * bandWidth_pd, _X1);

		_Y1 = _mm256_load_pd(pCache + 5 * bandWidth_pd);
		_mm256_store_pd(pOutput - bandWidth_pd, _Y1);

		i += hapinfo.position;
		i_r -= hapinfo.position - 1;
	}

	switch (hapinfo.position)
	{
	case 0:
		// -1
		_M1 = _mm256_setzero_pd();
		_X1 = _mm256_setzero_pd();
		_Y1 = _mm256_setzero_pd();

		pInput += 3 * bandWidth_pd;

		// 0
		_M0 = _mm256_setzero_pd();
		_mm256_store_pd(pOutput, _M0);
		_X0 = _mm256_setzero_pd();
		_mm256_store_pd(pOutput + bandWidth_pd, _X0);
		_Y0 = _mm256_setzero_pd();
		_mm256_store_pd(pOutput + 2 * bandWidth_pd, _Y0);

		pOutput += 3 * bandWidth_pd;

		// 1
		__m256d Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	case 2:
		// 2
		Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		// 3
		Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	default:
	{
		for (; i < COLS; i += 2)
		{
			// Even
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// Odd
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}

		if (COLS % 2 == 0)
		{
			// Last Element - Even
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + i_r));

			// -2
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -1
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -0
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}
		else
		{
			__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + i_r));

			// -2
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -1
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -0
			_mm256_shift_left_si256<8>(_hap);
			Distm0 = ComputeDistm_pd(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}
	}
	}
}

void prepareReadParams(Context<double> &ctx, readinfo &read, double fGapM, __m256i &_rs, __m256d &_distM, __m256d &_distN, const size_t row)
{
	ConvertChars_pd(read.rs + row, _rs);

	__m128i _q128 = _mm_set1_epi64x(*((const __int32*)(read.q + row)));
	__m256i _q = _mm256_cvtepi8_epi64(_q128);

	__m256i _qMasked = _mm256_and_si256(
		_q,
		_mm256_set1_epi64x(127)
	);

	__m256d _distmBase = _mm256_i64gather_pd(
		ctx.ph2pr,
		_qMasked,
		8);

	__m256d _fGapM = _mm256_set1_pd(fGapM);

	_distM = _mm256_mul_pd(
		_fGapM,
		_mm256_sub_pd(
			_mm256_set1_pd(1.0),
			_distmBase
		));

	_distN = _mm256_mul_pd(
		_fGapM,
		_mm256_div_pd(
			_distmBase,
			_mm256_set1_pd(3.0)
		));
}

void compute_prob_avxd(readinfo &read, vector<hapinfo> &hap_array)
{
	size_t numHaplotypes = hap_array.size();

	size_t COLS_MIN = 0, COLS_MAX = 0;

	computeHaplotypeSimilarities(hap_array, COLS_MIN, COLS_MAX);

	if (COLS_MIN < bandWidth_pd)
	{
		compute_prob_scalarf(read, hap_array);
		return;
	}

	Context<double> ctx;

	size_t ROWS = read.rslen + 1;

	double yInitial = ctx.INITIAL_CONSTANT / COLS_MAX;

	// Initialize memory
	size_t COLS_PADDED = COLS_MAX + 2 * bandWidth_pd;

	size_t hapSize = COLS_PADDED * sizeof(int64_t);
	size_t bandSize = 3 * bandWidth_pd * COLS_PADDED * sizeof(double);
	size_t columnSize = 6 * bandWidth_pd * ROWS * sizeof(double);

	int64_t* i_hap = (int64_t*)scalable_aligned_malloc(hapSize, 32);
	double* pRow0 = (double*)scalable_aligned_malloc(bandSize, 32);
	double* pRow1 = (double*)scalable_aligned_malloc(bandSize, 32);
	double* pColumnCache = (double*)scalable_aligned_malloc(columnSize, 32);

	double* pMM = (double*)scalable_aligned_malloc(sizeof(double) * read.rslen * 5, 32);
	double* pMX = pMM + read.rslen;
	double* pMY = pMX + read.rslen;
	double* pZZ = pMY + read.rslen;

	int _c = read.c[0] & 127;
	double fZZ = ctx.ph2pr[_c];
	double fGapM = 1.0f - fZZ;

	for (size_t r = 0; r < read.rslen; r++)
	{
		int _i = read.i[r] & 127;
		int _d = read.d[r] & 127;
		pMM[r] = ctx.set_mm_prob(_i, _d) / fGapM;
		pMX[r] = ctx.ph2pr[_i];
		pMY[r] = ctx.ph2pr[_d];
		pZZ[r] = fZZ;
	}

	for (size_t hap_idx = 0; hap_idx < numHaplotypes; ++hap_idx)
	{
#ifdef _UNIT_TEST_DUMP
		FILE* fTable = fopen("pairHmm_avx2.csv", "w");
		assert(fTable != NULL);
#endif

		hapinfo &haplotype = hap_array[hap_idx];

		size_t COLS = haplotype.haplen;

		size_t nIterations = COLS / bandWidth_pd;
		size_t nRemainder = COLS % bandWidth_pd;

		size_t col = 0;
		size_t inverseCol = COLS;
		__m256i _i_hap = _mm256_set1_epi32(4);
		char* hap = haplotype.hap;

		_mm256_storeu_si256((__m256i*) (i_hap + inverseCol), _i_hap);
		inverseCol -= bandWidth_pd;

		if (nRemainder > 0)
		{
			ConvertChars_pd(hap + col, _i_hap);

			_i_hap = _mm256_permute4x64_epi64(
				_i_hap,
				0x1B);
			_mm256_storeu_si256((__m256i*) (i_hap + inverseCol), _i_hap);

			col += nRemainder;
			inverseCol -= nRemainder;
		}

		while (nIterations-- > 0)
		{
			ConvertChars_pd(hap + col, _i_hap);

			_i_hap = _mm256_permute4x64_epi64(
				_i_hap,
				0x1B);
			_mm256_storeu_si256((__m256i*) (i_hap + inverseCol), _i_hap);

			col += bandWidth_pd;
			inverseCol -= bandWidth_pd;
		}

		bool bShareWithNext = (hap_idx + 1 < numHaplotypes) && (hap_array[hap_idx + 1].position > 0) && (haplotype.position <= hap_array[hap_idx + 1].position);

		size_t firstDistinct = haplotype.position;

		size_t nextDistinct = COLS;
		if (bShareWithNext)
		{
			nextDistinct = hap_array[hap_idx + 1].position;
		}

#ifdef _UNIT_TEST
		size_t COLS0 = haplotype.haplen + 1;

		double** M = DebugCompute(ctx, read, haplotype, yInitial);
		double** X = M + ROWS;
		double** Y = X + ROWS;

		double resultDebug = 0.0f;

		for (int c = 0; c < COLS0; c++)
		{
			resultDebug += M[ROWS - 1][c] + X[ROWS - 1][c];
		}

		resultDebug *= (double)COLS_MAX / (double)(haplotype.haplen);

#ifdef _UNIT_TEST_DUMP
		DebugDump(M, X, Y, ROWS, COLS0);
#endif // _UNIT_TEST_DUMP

#endif // _UNIT_TEST

		size_t row = 0;

		__m256i _rs;
		__m256d _distM, _distN;

		prepareReadParams(ctx, read, fGapM, _rs, _distM, _distN, row);

		nIterations = read.rslen / bandWidth_pd;
		nRemainder = read.rslen % bandWidth_pd;

		double* pColumnPtr = pColumnCache;

		compute_bulk_band_first(
			read, haplotype,
			_rs, _distM, _distN,
			i_hap,
			pMM,
			pRow0, yInitial, pColumnPtr);

#ifdef _UNIT_TEST
		{
			double* pStart = pRow0;

#ifdef _UNIT_TEST_DUMP
			fprintf(fTable, ", ");
			for (size_t cc = 0; cc < COLS0; cc++)
			{
				fprintf(fTable, "%zd, ", cc);
			}
			fprintf(fTable, "\n\n");

			for (size_t rr = 0; rr < ((nRemainder == 0) ? bandWidth_pd : nRemainder); rr++)
			{
				double* pInput = pStart + (3 * bandWidth_pd + 1) * rr;

				fprintf(fTable, "%zd, ", rr + 1);
				for (int cc = 0; cc < firstDistinct; cc++)
				{
					fprintf(fTable, ", ");
					pInput += 3 * bandWidth_pd;
				}
				for (size_t cc = firstDistinct; cc < COLS0; cc++)
				{
					fprintf(fTable, "%e, ", *pInput);
					pInput += 3 * bandWidth_pd;
				}
				fprintf(fTable, "\n");

				pInput = pStart + (3 * bandWidth_pd + 1) * rr + bandWidth_pd;

				fprintf(fTable, ", ");
				for (int cc = 0; cc < firstDistinct; cc++)
				{
					fprintf(fTable, ", ");
					pInput += 3 * bandWidth_pd;
				}
				for (size_t cc = firstDistinct; cc < COLS0; cc++)
				{
					fprintf(fTable, "%e, ", *pInput);
					pInput += 3 * bandWidth_pd;
				}
				fprintf(fTable, "\n");

				pInput = pStart + (3 * bandWidth_pd + 1) * rr + 2 * bandWidth_pd;

				fprintf(fTable, ", ");
				for (size_t cc = 0; cc < firstDistinct; cc++)
				{
					fprintf(fTable, ", ");
					pInput += 3 * bandWidth_pd;
				}
				for (size_t cc = firstDistinct; cc < COLS0; cc++)
				{
					fprintf(fTable, "%e, ", *pInput);
					pInput += 3 * bandWidth_pd;
				}
				fprintf(fTable, "\n\n");
				fflush(fTable);
			}

			pStart = pRow0;
#endif //_UNIT_TEST_DUMP

			for (size_t i = 0; i < ((nRemainder == 0) ? bandWidth_pd : nRemainder); i++)
			{
				size_t r = row + i + 1;

				double* pInput = pStart + 3 * bandWidth_pd * firstDistinct;

				for (size_t c = firstDistinct; c < COLS0; c++)
				{
					DebugAssertClose(pInput[0], M[r][c]);
					DebugAssertClose(pInput[bandWidth_pd], X[r][c]);
					DebugAssertClose(pInput[2 * bandWidth_pd], Y[r][c]);

					pInput += 3 * bandWidth_pd;
				}

				pStart += (3 * bandWidth_pd + 1);
			}
		}
#endif //_UNIT_TEST

		if (nRemainder == 0)
		{
			nIterations--;

			row += bandWidth_pd;
		}
		else
		{
			size_t offset = (3 * bandWidth_pd + 1) * (bandWidth_pd - nRemainder);

			// Copy correct row to row 0
			double* pOutput = pRow0 + 3 * bandWidth_pd * COLS + 3 * bandWidth_pd * bandWidth_pd - 1;

			size_t firstCopied = max(firstDistinct, 4ull);

			for (size_t c = COLS + bandWidth_pd; c-- > firstCopied; )
			{
				*pOutput = *(pOutput - offset);
				*(pOutput - bandWidth_pd) = *(pOutput - offset - bandWidth_pd);
				*(pOutput - 2 * bandWidth_pd) = *(pOutput - offset - 2 * bandWidth_pd);

				pOutput -= 3 * bandWidth_pd;
			}

			row += nRemainder;
#ifdef _UNIT_TEST_DUMP
			{
				double* pStart = pRow0;

				//				size_t rr = bandWidth_pd - 1;
				for (size_t rr = 0; rr < bandWidth_pd; rr++)
				{
					double* pInput = pStart + (3 * bandWidth_pd + 1) * rr;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < COLS0; cc++)
					{
						fprintf(fTable, "%e, ", *pInput);
						pInput += 3 * bandWidth_pd;
					}
					fprintf(fTable, "\n");

					pInput = pStart + (3 * bandWidth_pd + 1) * rr + bandWidth_pd;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < COLS0; cc++)
					{
						fprintf(fTable, "%e, ", *pInput);
						pInput += 3 * bandWidth_pd;
					}
					fprintf(fTable, "\n");

					pInput = pStart + (3 * bandWidth_pd + 1) * rr + 2 * bandWidth_pd;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < COLS0; cc++)
					{
						fprintf(fTable, "%e, ", *pInput);
						pInput += 3 * bandWidth_pd;
					}
					fprintf(fTable, "\n\n");
					fflush(fTable);
				}
			}
#endif //_UNIT_TEST_DUMP
		}

		if (bShareWithNext)
		{
			memcpy(pColumnPtr, pRow0 + 3 * bandWidth_pd * (nextDistinct - 2), 6 * bandWidth_pd * sizeof(double));
		}

		pColumnPtr += 6 * bandWidth_pd;

		while (nIterations-- > 0)
		{
			prepareReadParams(ctx, read, fGapM, _rs, _distM, _distN, row);

			compute_bulk_band(
				read, haplotype,
				_rs, _distM, _distN,
				i_hap,
				pMM,
				pRow0, pRow1, pColumnPtr,
				row);

			double* pTemp = pRow1; pRow1 = pRow0; pRow0 = pTemp;

			if (bShareWithNext)
			{
				memcpy(pColumnPtr, pRow0 + 3 * bandWidth_pd * (nextDistinct - 2), 6 * bandWidth_pd * sizeof(double));
			}

			pColumnPtr += 6 * bandWidth_pd;

#ifdef _UNIT_TEST
			{
				double* pStart = pRow0;

#ifdef _UNIT_TEST_DUMP
				for (size_t rr = 0; rr < bandWidth_pd; rr++)
				{
					double* pInput = pStart + (3 * bandWidth_pd + 1) * rr;

					fprintf(fTable, "%zd, ", rr + row + 1);
					for (size_t cc = 0; cc < firstDistinct; cc++)
					{
						fprintf(fTable, ", ");
						pInput += 3 * bandWidth_pd;
					}
					for (size_t cc = firstDistinct; cc < COLS0; cc++)
					{
						fprintf(fTable, "%e, ", *pInput);
						pInput += 3 * bandWidth_pd;
					}
					fprintf(fTable, "\n");

					pInput = pStart + (3 * bandWidth_pd + 1) * rr + bandWidth_pd;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < firstDistinct; cc++)
					{
						fprintf(fTable, ", ");
						pInput += 3 * bandWidth_pd;
					}
					for (size_t cc = firstDistinct; cc < COLS0; cc++)
					{
						fprintf(fTable, "%e, ", *pInput);
						pInput += 3 * bandWidth_pd;
					}
					fprintf(fTable, "\n");

					pInput = pStart + (3 * bandWidth_pd + 1) * rr + 2 * bandWidth_pd;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < firstDistinct; cc++)
					{
						fprintf(fTable, ", ");
						pInput += 3 * bandWidth_pd;
					}
					for (size_t cc = firstDistinct; cc < COLS0; cc++)
					{
						fprintf(fTable, "%e, ", *pInput);
						pInput += 3 * bandWidth_pd;
					}
					fprintf(fTable, "\n\n");
				}
				fflush(fTable);

				pStart = pRow0;
#endif //_UNIT_TEST_DUMP

				for (size_t i = 0; i < bandWidth_pd; i++)
				{
					size_t r = row + i + 1;

					double* pInput = pStart + 3 * bandWidth_pd * firstDistinct;

					for (size_t c = firstDistinct; c < COLS0; c++)
					{
						DebugAssertClose(pInput[0], M[r][c]);
						DebugAssertClose(pInput[bandWidth_pd], X[r][c]);
						DebugAssertClose(pInput[2 * bandWidth_pd], Y[r][c]);

						pInput += 3 * bandWidth_pd;
					}

					pStart += (3 * bandWidth_pd + 1);
				}
			}
#endif //_UNIT_TEST

			row += bandWidth_pd;
		}

		double* pInput = pRow0 + 3 * bandWidth_pd * (firstDistinct + bandWidth_pd - 1);

		double result = haplotype.score;

		size_t rc = firstDistinct;
		for (; rc < nextDistinct; rc++)
		{
#ifdef _UNIT_TEST
			DebugAssertClose(M[ROWS - 1][rc], pInput[bandWidth_pd - 1]);
			DebugAssertClose(X[ROWS - 1][rc], pInput[2 * bandWidth_pd - 1]);
			DebugAssertClose(Y[ROWS - 1][rc], pInput[3 * bandWidth_pd - 1]);
#endif //_UNIT_TEST

			result += pInput[bandWidth_pd - 1] + pInput[2 * bandWidth_pd - 1];
			pInput += 3 * bandWidth_pd;
		}

		if (bShareWithNext)
		{
			hap_array[hap_idx + 1].score = result;
		}

		for (; rc <= COLS; rc++)
		{

#ifdef _UNIT_TEST
			DebugAssertClose(M[ROWS - 1][rc], pInput[bandWidth_pd - 1]);
			DebugAssertClose(X[ROWS - 1][rc], pInput[2 * bandWidth_pd - 1]);
			DebugAssertClose(Y[ROWS - 1][rc], pInput[3 * bandWidth_pd - 1]);
#endif //_UNIT_TEST

			result += pInput[bandWidth_pd - 1] + pInput[2 * bandWidth_pd - 1];
			pInput += 3 * bandWidth_pd;
		}

		haplotype.score = result * (double)COLS_MAX / (double)(haplotype.haplen);

#ifdef _UNIT_TEST
		DebugAssertClose(resultDebug, double(haplotype.score));

		delete[] M[0];
		delete[] M;

#ifdef _UNIT_TEST_DUMP
		fclose(fTable);
#endif //_UNIT_TEST_DUMP
#endif //_UNIT_TEST
	}

	// Free memory
	scalable_free(pColumnCache);
	scalable_free(pRow1);
	scalable_free(pRow0);
	scalable_free(i_hap);

	scalable_free(pMM);
}

