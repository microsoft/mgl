#include "compute_prob_scalar.h"
#include "compute_prob_avx.h"

using namespace std;

const size_t bandWidth_ps = 8;

template <size_t __N>
inline void _mm256_shift_left_si256(__m256i& a, __m256i b) {
	__m256i c = _mm256_permute2x128_si256(a, b, 0x03);
	a = _mm256_alignr_epi8(a, c, 16 - __N);
}

template <size_t __N>
inline void _mm256_shift_left_si256(__m256i& a) {
	__m256i c = _mm256_permute2x128_si256(a, _mm256_setzero_si256(), 0x03);
	a = _mm256_alignr_epi8(a, c, 16 - __N);
}

inline __m256 _mm256_shift_left_ps(__m256 a, const __m256 b) {
	__m256i c = _mm256_permute2x128_si256(
		_mm256_castps_si256(a),
		_mm256_castps_si256(b),
		0x03);

	return _mm256_castsi256_ps(
		_mm256_alignr_epi8(
			_mm256_castps_si256(a),
			c,
			12));
}

inline void _mm256_shift_left_ps(__m256& a) {
	a = _mm256_shift_left_ps(a, _mm256_setzero_ps());
}

inline void ConvertChars_ps(const char* rs, __m256i &_rs)
{
	__m128i _rs128 = _mm_set1_epi64x((*((const int64_t*)rs)) & 0x0f0f0f0f0f0f0f0fll);

	__m128i _rsConverted = _mm_shuffle_epi8(
		_mm_set_epi8(0, 4, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 0),
		_rs128);

	_rs = _mm256_cvtepi8_epi32(_rsConverted);
}

inline __m256 ComputeDistm_ps(const __m256i _rs, const __m256 _distM, const __m256 _distN, __m256i _hap)
{
	__m256i _match = _mm256_or_si256(
		_mm256_cmpeq_epi32(_rs, _hap),
		_mm256_slli_epi32(
			_mm256_or_si256(_rs, _hap),
			29));

	return _mm256_blendv_ps(
		_distN,
		_distM,
		_mm256_castsi256_ps(_match)
	);
}

inline __m256 ComputeDistm_ps(const __m256i _rs, const __m256 _distM, const __m256 _distN, int32_t* i_hap, size_t col)
{
	__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + col));

	return ComputeDistm_ps(_rs, _distM, _distN, _hap);
}

inline void AdvanceCellFirstRow(__m256 &_M0, __m256 &_X0, __m256 &_Y0, __m256 &_M1, __m256 &_X1, __m256 &_Y1, __m256 &_YInitial, float * &pOutput, __m256 &Distm0, __m256 &_pMM, __m256 &_pMX, __m256 &_pMY, __m256 &_pZZ)
{
	_M0 = _mm256_mul_ps(
		Distm0,
		_mm256_fmadd_ps(
			_M0,
			_pMM,
			_mm256_add_ps(
				_X0,
				_Y0
			)));

	_mm256_store_ps(pOutput, _M0);

	_Y0 = _mm256_fmadd_ps(
		_M1,
		_pMY,
		_mm256_mul_ps(
			_Y1,
			_pZZ
		));

	_mm256_shift_left_ps(_M1);

	_mm256_shift_left_ps(_X1);

	_X0 = _mm256_fmadd_ps(
		_M1,
		_pMX,
		_mm256_mul_ps(
			_X1,
			_pZZ
		));

	_mm256_store_ps(pOutput + bandWidth_ps, _X0);

	_Y1 = _mm256_shift_left_ps(_Y1, _YInitial);

	_mm256_store_ps(pOutput + 2 * bandWidth_ps, _Y0);

	pOutput += 3 * bandWidth_ps;
}


inline void AdvanceCell(__m256 &_M0, __m256 &_X0, __m256 &_Y0, __m256 &_M1, __m256 &_X1, __m256 &_Y1, float * &pInput, float * &pOutput, __m256 &Distm0, __m256 &_pMM, __m256 &_pMX, __m256 &_pMY, __m256 &_pZZ)
{
	_M0 = _mm256_mul_ps(
		Distm0,
		_mm256_fmadd_ps(
			_M0,
			_pMM,
			_mm256_add_ps(
				_X0,
				_Y0
			)));

	_mm256_store_ps(pOutput, _M0);

	_Y0 = _mm256_fmadd_ps(
		_M1,
		_pMY,
		_mm256_mul_ps(
			_Y1,
			_pZZ
		));

	_M1 = _mm256_shift_left_ps(_M1, _mm256_load_ps(pInput));

	_X1 = _mm256_shift_left_ps(_X1, _mm256_load_ps(pInput + bandWidth_ps));

	_X0 = _mm256_fmadd_ps(
		_M1,
		_pMX,
		_mm256_mul_ps(
			_X1,
			_pZZ
		));

	_mm256_store_ps(pOutput + bandWidth_ps, _X0);

	_Y1 = _mm256_shift_left_ps(_Y1, _mm256_load_ps(pInput + 2 * bandWidth_ps));

	_mm256_store_ps(pOutput + 2 * bandWidth_ps, _Y0);

	pInput += 3 * bandWidth_ps;
	pOutput += 3 * bandWidth_ps;
}


void compute_bulk_band_first(
	readinfo& read, hapinfo &hapinfo,
	const __m256i _rs, const __m256 _distM, const __m256 _distN,
	int32_t* i_hap,
	float* pMM,
	float* pRow0, float yInitial, float* pCache)
{
	int ROWS = read.rslen;

	float* pMX = pMM + ROWS;
	float* pMY = pMX + ROWS;
	float* pZZ = pMY + ROWS;

	__m256 _YInitial = _mm256_set1_ps(yInitial);

	size_t COLS = hapinfo.haplen;

	float* pOutput = pRow0;

	__m256 _pMM = _mm256_loadu_ps(pMM);

	__m256 _pMX = _mm256_loadu_ps(pMX);

	__m256 _pMY = _mm256_loadu_ps(pMY);

	__m256 _pZZ = _mm256_set1_ps(*pZZ);

	__m256 _M0, _X0, _Y0;
	__m256 _M1, _X1, _Y1;

	size_t i = 0;
	size_t i_r = COLS;

	if (hapinfo.position > 0)
	{
		size_t offset = 3 * bandWidth_ps * hapinfo.position;

		pOutput += offset;

		_M0 = _mm256_load_ps(pCache);
		_mm256_store_ps(pOutput - 6 * bandWidth_ps, _M0);
		_mm256_shift_left_ps(_M0);

		_X0 = _mm256_load_ps(pCache + bandWidth_ps);
		_mm256_store_ps(pOutput - 5 * bandWidth_ps, _X0);
		_mm256_shift_left_ps(_X0);

		_Y0 = _mm256_load_ps(pCache + 2 * bandWidth_ps);
		_mm256_store_ps(pOutput - 4 * bandWidth_ps, _Y0);
		_Y0 = _mm256_shift_left_ps(_Y0, _YInitial);


		_M1 = _mm256_load_ps(pCache + 3 * bandWidth_ps);
		_mm256_store_ps(pOutput - 3 * bandWidth_ps, _M1);

		_X1 = _mm256_load_ps(pCache + 4 * bandWidth_ps);
		_mm256_store_ps(pOutput - 2 * bandWidth_ps, _X1);

		_Y1 = _mm256_load_ps(pCache + 5 * bandWidth_ps);
		_mm256_store_ps(pOutput - bandWidth_ps, _Y1);

		i += hapinfo.position;
		i_r -= hapinfo.position - 1;
	}
	__m256 Distm0;

	switch (hapinfo.position)
	{
	case 0:

		// -1
		_M1 = _mm256_setzero_ps();
		_X1 = _mm256_setzero_ps();
		_Y1 = _mm256_blend_ps(_YInitial, _mm256_setzero_ps(), 0xFE);

		// 0
		_M0 = _mm256_setzero_ps();
		_mm256_store_ps(pOutput, _M0);
		_X0 = _mm256_setzero_ps();
		_mm256_store_ps(pOutput + bandWidth_ps, _X0);
		_Y0 = _mm256_setzero_ps();
		_mm256_store_ps(pOutput + 2 * bandWidth_ps, _Y0);

		pOutput += 3 * bandWidth_ps;

		// 1
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	case 2:
		// 2
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		// 3
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	case 4:
		// 4
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		// 5
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	case 6:
		// 6
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		// 7
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	default:
	{
		for (; i < COLS; i += 2)
		{
			// Even
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// Odd
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}

		if (COLS % 2 == 0)
		{
			// Last Element - Even
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + i_r));

			// -6
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -5
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -4
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -3
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -2
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -1
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -0
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}
		else
		{
			__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + i_r));

			// -6
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -5
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -4
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -3
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -2
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -1
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M1, _X1, _Y1, _M0, _X0, _Y0, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -0
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCellFirstRow(_M0, _X0, _Y0, _M1, _X1, _Y1, _YInitial, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}
	}
	}
}

void compute_bulk_band(
	readinfo &read, hapinfo &hapinfo,
	const __m256i _rs, const __m256 _distM, const __m256 _distN,
	int32_t* i_hap,
	float * pMM,
	float* pRow0, float* pRow1, float* pCache,
	size_t row)
{
	int ROWS = read.rslen;

	float* pMX = pMM + ROWS;
	float* pMY = pMX + ROWS;
	float* pZZ = pMY + ROWS;

	size_t COLS = hapinfo.haplen;

	// Skip forward triangle
	float* pInput = pRow0 + 3 * bandWidth_ps * (bandWidth_ps - 1);
	float* pOutput = pRow1;

	__m256 _pMM = _mm256_loadu_ps(pMM + row);

	__m256 _pMX = _mm256_loadu_ps(pMX + row);

	__m256 _pMY = _mm256_loadu_ps(pMY + row);

	__m256 _pZZ = _mm256_set1_ps(*pZZ);

	__m256 _M0, _X0, _Y0;
	__m256 _M1, _X1, _Y1;

	size_t i = 0;
	size_t i_r = COLS;

	if (hapinfo.position > 0)
	{
		size_t offset = 3 * bandWidth_ps * hapinfo.position;

		pOutput += offset;
		pInput += offset;

		_M0 = _mm256_load_ps(pCache);
		_mm256_store_ps(pOutput - 6 * bandWidth_ps, _M0);
		_M0 = _mm256_shift_left_ps(_M0, _mm256_load_ps(pInput - 3 * bandWidth_ps));

		_X0 = _mm256_load_ps(pCache + bandWidth_ps);
		_mm256_store_ps(pOutput - 5 * bandWidth_ps, _X0);
		_X0 = _mm256_shift_left_ps(_X0, _mm256_load_ps(pInput - 2 * bandWidth_ps));

		_Y0 = _mm256_load_ps(pCache + 2 * bandWidth_ps);
		_mm256_store_ps(pOutput - 4 * bandWidth_ps, _Y0);
		_Y0 = _mm256_shift_left_ps(_Y0, _mm256_load_ps(pInput - bandWidth_ps));


		_M1 = _mm256_load_ps(pCache + 3 * bandWidth_ps);
		_mm256_store_ps(pOutput - 3 * bandWidth_ps, _M1);

		_X1 = _mm256_load_ps(pCache + 4 * bandWidth_ps);
		_mm256_store_ps(pOutput - 2 * bandWidth_ps, _X1);

		_Y1 = _mm256_load_ps(pCache + 5 * bandWidth_ps);
		_mm256_store_ps(pOutput - bandWidth_ps, _Y1);

		i += hapinfo.position;
		i_r -= hapinfo.position - 1;
	}

	__m256 Distm0;

	switch (hapinfo.position)
	{
	case 0:
		// -1
		_M1 = _mm256_setzero_ps();
		_X1 = _mm256_setzero_ps();
		_Y1 = _mm256_setzero_ps();

		pInput += 3 * bandWidth_ps;

		// 0
		_M0 = _mm256_setzero_ps();
		_mm256_store_ps(pOutput, _M0);
		_X0 = _mm256_setzero_ps();
		_mm256_store_ps(pOutput + bandWidth_ps, _X0);
		_Y0 = _mm256_setzero_ps();
		_mm256_store_ps(pOutput + 2 * bandWidth_ps, _Y0);

		pOutput += 3 * bandWidth_ps;

		// 1
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	case 2:
		// 2
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		// 3
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	case 4:
		// 4
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		// 5
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	case 6:
		// 6
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		// 7
		Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
		AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

		i += 2;

	default:
	{
		for (; i < COLS; i += 2)
		{
			// Even
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// Odd
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}

		if (COLS % 2 == 0)
		{
			// Last Element - Even
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, i_hap, --i_r);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + i_r));

			// -6
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -5
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -4
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -3
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -2
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -1
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -0
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}
		else
		{
			__m256i _hap = _mm256_loadu_si256((const __m256i*)(i_hap + i_r));

			// -6
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -5
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -4
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -3
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -2
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -1
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M1, _X1, _Y1, _M0, _X0, _Y0, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);

			// -0
			_mm256_shift_left_si256<4>(_hap);
			Distm0 = ComputeDistm_ps(_rs, _distM, _distN, _hap);
			AdvanceCell(_M0, _X0, _Y0, _M1, _X1, _Y1, pInput, pOutput, Distm0, _pMM, _pMX, _pMY, _pZZ);
		}
	}
	}
}

void prepareReadParams(Context<float> &ctx, readinfo &read, float fGapM, __m256i &_rs, __m256 &_distM, __m256 &_distN, const size_t row)
{
	ConvertChars_ps(read.rs + row, _rs);

	__m128i _q128 = _mm_set1_epi64x(*((const int64_t*)(read.q + row)));
	__m256i _q = _mm256_cvtepi8_epi32(_q128);

	__m256i _qMasked = _mm256_and_si256(
		_q,
		_mm256_set1_epi32(127)
	);

	__m256 _distmBase = _mm256_i32gather_ps(
		ctx.ph2pr,
		_qMasked,
		4);

	__m256 _fGapM = _mm256_set1_ps(fGapM);

	_distM = _mm256_mul_ps(
		_fGapM,
		_mm256_sub_ps(
			_mm256_set1_ps(1.0f),
			_distmBase
		));

	_distN = _mm256_mul_ps(
		_fGapM,
		_mm256_div_ps(
			_distmBase,
			_mm256_set1_ps(3.0f)
		));
}

void compute_prob_avxf(readinfo &read, vector<hapinfo> &hap_array)
{
	size_t numHaplotypes = hap_array.size();

	size_t COLS_MIN = 0, COLS_MAX = 0;

	computeHaplotypeSimilarities(hap_array, COLS_MIN, COLS_MAX);

	if (COLS_MIN < bandWidth_ps)
	{
		compute_prob_scalarf(read, hap_array);
		return;
	}

	Context<float> ctx;

	size_t ROWS = read.rslen + 1;

	float yInitial = ctx.INITIAL_CONSTANT / COLS_MAX;

	// Initialize memory
	size_t COLS_PADDED = COLS_MAX + 2 * bandWidth_ps;

	size_t hapSize = COLS_PADDED * sizeof(int32_t);
	size_t bandSize = 3 * bandWidth_ps * COLS_PADDED * sizeof(float);
	size_t columnSize = 6 * bandWidth_ps * ROWS * sizeof(float);

	int32_t* i_hap = (int32_t*)scalable_aligned_malloc(hapSize, 32);
	float* pRow0 = (float*)scalable_aligned_malloc(bandSize, 32);
	float* pRow1 = (float*)scalable_aligned_malloc(bandSize, 32);
	float* pColumnCache = (float*)scalable_aligned_malloc(columnSize, 32);

	float* pMM = (float*)scalable_aligned_malloc(sizeof(float) * read.rslen * 5, 32);
	float* pMX = pMM + read.rslen;
	float* pMY = pMX + read.rslen;
	float* pZZ = pMY + read.rslen;

	int _c = read.c[0] & 127;
	float fZZ = ctx.ph2pr[_c];
	float fGapM = 1.0f - fZZ;

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

		size_t nIterations = COLS / bandWidth_ps;
		size_t nRemainder = COLS % bandWidth_ps;

		size_t col = 0;
		size_t inverseCol = COLS;
		__m256i _i_hap = _mm256_set1_epi32(4);
		char* hap = haplotype.hap;

		_mm256_storeu_si256((__m256i*) (i_hap + inverseCol), _i_hap);
		inverseCol -= bandWidth_ps;

		if (nRemainder > 0)
		{
			ConvertChars_ps(hap + col, _i_hap);

			_i_hap = _mm256_permutevar8x32_epi32(
				_i_hap,
				_mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
			_mm256_storeu_si256((__m256i*) (i_hap + inverseCol), _i_hap);

			col += nRemainder;
			inverseCol -= nRemainder;
		}

		while (nIterations-- > 0)
		{
			ConvertChars_ps(hap + col, _i_hap);

			_i_hap = _mm256_permutevar8x32_epi32(
				_i_hap,
				_mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
			_mm256_storeu_si256((__m256i*) (i_hap + inverseCol), _i_hap);

			col += bandWidth_ps;
			inverseCol -= bandWidth_ps;
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

		float** M = DebugCompute(ctx, read, haplotype, yInitial);
		float** X = M + ROWS;
		float** Y = X + ROWS;

		double resultDebug = 0.0;

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
		__m256 _distM, _distN;

		prepareReadParams(ctx, read, fGapM, _rs, _distM, _distN, row);

		nIterations = read.rslen / bandWidth_ps;
		nRemainder = read.rslen % bandWidth_ps;

		float* pColumnPtr = pColumnCache;

		compute_bulk_band_first(
			read, haplotype,
			_rs, _distM, _distN,
			i_hap,
			pMM,
			pRow0, yInitial, pColumnPtr);

#ifdef _UNIT_TEST
		{
			float* pStart = pRow0;

#ifdef _UNIT_TEST_DUMP
			fprintf(fTable, ", ");
			for (size_t cc = 0; cc < COLS0; cc++)
			{
				fprintf(fTable, "%zd, ", cc);
			}
			fprintf(fTable, "\n\n");

			for (size_t rr = 0; rr < ((nRemainder == 0) ? bandWidth_ps : nRemainder); rr++)
			{
				float* pInput = pStart + (3 * bandWidth_ps + 1) * rr;

				fprintf(fTable, "%zd, ", rr + 1);
				for (size_t cc = 0; cc < firstDistinct; cc++)
				{
					fprintf(fTable, ", ");
					pInput += 3 * bandWidth_ps;
				}
				for (size_t cc = firstDistinct; cc < COLS0; cc++)
				{
					fprintf(fTable, "%f, ", *pInput);
					pInput += 3 * bandWidth_ps;
				}
				fprintf(fTable, "\n");

				pInput = pStart + (3 * bandWidth_ps + 1) * rr + bandWidth_ps;

				fprintf(fTable, ", ");
				for (size_t cc = 0; cc < firstDistinct; cc++)
				{
					fprintf(fTable, ", ");
					pInput += 3 * bandWidth_ps;
				}
				for (size_t cc = firstDistinct; cc < COLS0; cc++)
				{
					fprintf(fTable, "%f, ", *pInput);
					pInput += 3 * bandWidth_ps;
				}
				fprintf(fTable, "\n");

				pInput = pStart + (3 * bandWidth_ps + 1) * rr + 2 * bandWidth_ps;

				fprintf(fTable, ", ");
				for (size_t cc = 0; cc < firstDistinct; cc++)
				{
					fprintf(fTable, ", ");
					pInput += 3 * bandWidth_ps;
				}
				for (size_t cc = firstDistinct; cc < COLS0; cc++)
				{
					fprintf(fTable, "%f, ", *pInput);
					pInput += 3 * bandWidth_ps;
				}
				fprintf(fTable, "\n\n");
				fflush(fTable);
			}

			pStart = pRow0;
#endif //_UNIT_TEST_DUMP

			for (size_t i = 0; i < ((nRemainder == 0) ? bandWidth_ps : nRemainder); i++)
			{
				size_t r = row + i + 1;

				float* pInput = pStart + 3 * bandWidth_ps * firstDistinct;

				for (size_t c = firstDistinct; c < COLS0; c++)
				{
					DebugAssertClose(pInput[0], M[r][c]);
					DebugAssertClose(pInput[bandWidth_ps], X[r][c]);
					DebugAssertClose(pInput[2 * bandWidth_ps], Y[r][c]);

					pInput += 3 * bandWidth_ps;
				}

				pStart += (3 * bandWidth_ps + 1);
			}
		}
#endif //_UNIT_TEST

		if (nRemainder == 0)
		{
			nIterations--;

			row += bandWidth_ps;
		}
		else
		{
			size_t offset = (3 * bandWidth_ps + 1) * (bandWidth_ps - nRemainder);

			// Copy correct row to row 0
			float* pOutput = pRow0 + 3 * bandWidth_ps * COLS + 3 * bandWidth_ps * bandWidth_ps - 1;

			size_t firstCopied = max(firstDistinct, 8ul);

			for (size_t c = COLS + bandWidth_ps; c-- > firstCopied; )
			{
				*pOutput = *(pOutput - offset);
				*(pOutput - bandWidth_ps) = *(pOutput - offset - bandWidth_ps);
				*(pOutput - 2 * bandWidth_ps) = *(pOutput - offset - 2 * bandWidth_ps);

				pOutput -= 3 * bandWidth_ps;
			}
			
			row += nRemainder;
#ifdef _UNIT_TEST_DUMP
			{
				float* pStart = pRow0;

				size_t rr = bandWidth_ps - 1;
				{
					float* pInput = pStart + (3 * bandWidth_ps + 1) * rr;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < COLS0; cc++)
					{
						fprintf(fTable, "%f, ", *pInput);
						pInput += 3 * bandWidth_ps;
					}
					fprintf(fTable, "\n");

					pInput = pStart + (3 * bandWidth_ps + 1) * rr + bandWidth_ps;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < COLS0; cc++)
					{
						fprintf(fTable, "%f, ", *pInput);
						pInput += 3 * bandWidth_ps;
					}
					fprintf(fTable, "\n");

					pInput = pStart + (3 * bandWidth_ps + 1) * rr + 2 * bandWidth_ps;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < COLS0; cc++)
					{
						fprintf(fTable, "%f, ", *pInput);
						pInput += 3 * bandWidth_ps;
					}
					fprintf(fTable, "\n\n");
					fflush(fTable);
				}
			}
#endif //_UNIT_TEST_DUMP
		}

		if (bShareWithNext)
		{
			memcpy(pColumnPtr, pRow0 + 3 * bandWidth_ps * (nextDistinct - 2), 6 * bandWidth_ps * sizeof(float));
		}

		pColumnPtr += 6 * bandWidth_ps;

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

			float* pTemp = pRow1; pRow1 = pRow0; pRow0 = pTemp;

			if (bShareWithNext)
			{
				memcpy(pColumnPtr, pRow0 + 3 * bandWidth_ps * (nextDistinct - 2), 6 * bandWidth_ps * sizeof(float));
			}

			pColumnPtr += 6 * bandWidth_ps;

#ifdef _UNIT_TEST
			{
				float* pStart = pRow0;

#ifdef _UNIT_TEST_DUMP
				for (size_t rr = 0; rr < bandWidth_ps; rr++)
				{
					float* pInput = pStart + (3 * bandWidth_ps + 1) * rr;

					fprintf(fTable, "%zd, ", rr + row + 1);
					for (size_t cc = 0; cc < firstDistinct; cc++)
					{
						fprintf(fTable, ", ");
						pInput += 3 * bandWidth_ps;
					}
					for (size_t cc = firstDistinct; cc < COLS0; cc++)
					{
						fprintf(fTable, "%f, ", *pInput);
						pInput += 3 * bandWidth_ps;
					}
					fprintf(fTable, "\n");

					pInput = pStart + (3 * bandWidth_ps + 1) * rr + bandWidth_ps;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < firstDistinct; cc++)
					{
						fprintf(fTable, ", ");
						pInput += 3 * bandWidth_ps;
					}
					for (size_t cc = firstDistinct; cc < COLS0; cc++)
					{
						fprintf(fTable, "%f, ", *pInput);
						pInput += 3 * bandWidth_ps;
					}
					fprintf(fTable, "\n");

					pInput = pStart + (3 * bandWidth_ps + 1) * rr + 2 * bandWidth_ps;

					fprintf(fTable, ", ");
					for (size_t cc = 0; cc < firstDistinct; cc++)
					{
						fprintf(fTable, ", ");
						pInput += 3 * bandWidth_ps;
					}
					for (size_t cc = firstDistinct; cc < COLS0; cc++)
					{
						fprintf(fTable, "%f, ", *pInput);
						pInput += 3 * bandWidth_ps;
					}
					fprintf(fTable, "\n\n");
				}
				fflush(fTable);

				pStart = pRow0;
#endif //_UNIT_TEST_DUMP

				for (size_t i = 0; i < bandWidth_ps; i++)
				{
					size_t r = row + i + 1;

					float* pInput = pStart + 3 * bandWidth_ps * firstDistinct;

					for (size_t c = firstDistinct; c < COLS0; c++)
					{
						DebugAssertClose(pInput[0], M[r][c]);
						DebugAssertClose(pInput[bandWidth_ps], X[r][c]);
						DebugAssertClose(pInput[2 * bandWidth_ps], Y[r][c]);

						pInput += 3 * bandWidth_ps;
					}

					pStart += (3 * bandWidth_ps + 1);
				}
			}
#endif //_UNIT_TEST

			row += bandWidth_ps;
		}

		float* pInput = pRow0 + 3 * bandWidth_ps * (firstDistinct + bandWidth_ps - 1);

		double result = haplotype.score;

		size_t rc = firstDistinct;
		for (; rc < nextDistinct; rc++)
		{
#ifdef _UNIT_TEST
			DebugAssertClose(M[ROWS - 1][rc], pInput[bandWidth_ps - 1]);
			DebugAssertClose(X[ROWS - 1][rc], pInput[2 * bandWidth_ps - 1]);
			DebugAssertClose(Y[ROWS - 1][rc], pInput[3 * bandWidth_ps - 1]);
#endif //_UNIT_TEST

			result += pInput[bandWidth_ps - 1] + pInput[2 * bandWidth_ps - 1];
			pInput += 3 * bandWidth_ps;
		}

		if (bShareWithNext)
		{
			hap_array[hap_idx + 1].score = result;
		}

		for (; rc <= COLS; rc++)
		{

#ifdef _UNIT_TEST
			DebugAssertClose(M[ROWS - 1][rc], pInput[bandWidth_ps - 1]);
			DebugAssertClose(X[ROWS - 1][rc], pInput[2 * bandWidth_ps - 1]);
			DebugAssertClose(Y[ROWS - 1][rc], pInput[3 * bandWidth_ps - 1]);
#endif //_UNIT_TEST

			result += pInput[bandWidth_ps - 1] + pInput[2 * bandWidth_ps - 1];
			pInput += 3 * bandWidth_ps;
		}

		haplotype.score = result * (float)COLS_MAX / (float)(haplotype.haplen);

#ifdef _UNIT_TEST
		DebugAssertClose(resultDebug, haplotype.score);

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

