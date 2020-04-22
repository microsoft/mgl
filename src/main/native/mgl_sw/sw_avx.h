#ifndef SW_AVX_H
#define SW_AVX_H

#include"sw_common.h"

int align_avx(const char *tseq, int target_length, const char *qseq, int query_length, swParameters parameters, int strategy, std::string * result_cigar);
void calculateMatrix_avx(int *target, int target_length, int *query, int query_length, int *bcktrack, int band_count, int default_bw, int actual_bw, int *score, int *step, int *gap, swParameters parameters, int overhangStrategy, ScoreMax *ez);
int calculateCigar_avx(int *bcktrack, int n, int m, int bandwidth, int overhangStrategy, ScoreMax *ez, std::string * cigar);



__forceinline int _mm256_get_epi32(__m256i a, int idx)
{
	__m128i vidx = _mm_cvtsi32_si128(idx);
	__m256i vidx256 = _mm256_castsi128_si256(vidx);
	__m256i  shuffled = _mm256_permutevar8x32_epi32(a, vidx256);
	return _mm256_cvtsi256_si32(shuffled);
};

template <size_t __N>
__forceinline __m256i _mm256_shift_left_si256(__m256i a, __m256i b) {
	__m256i c = _mm256_permute2x128_si256(a, b, 0x03);
	return _mm256_alignr_epi8(a, c, 16 - __N);
};

template <size_t __N>
__forceinline __m256i _mm256_shift_left_si256(__m256i a) {
	__m256i c = _mm256_permute2x128_si256(a, _mm256_setzero_si256(), 0x03);
	return _mm256_alignr_epi8(a, c, 16 - __N);
};

// returns position of [i, j] array element it array is stored in anti-diagonal bands
inline int bcktrMatrix_index(int i, int j, int n_col, int bw)
{
	int band = i / bw, J = i % bw, I = j + J;

	int pos = band * bw *n_col + I * bw + J;

	return pos;
}

#ifdef DEBUG_PRINT
static void print_bcktrMatrix(int *a, int n, int m, int bw)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
			fprintf(stderr, "%d\t", a[bcktrMatrix_index(i, j, m + bw - 1, bw)]);
		//		fprintf(stderr, "[%d, %d]%d:%d\t", i, j, bcktrMatrix_index(i, j, m + bw - 1, bw), a[bcktrMatrix_index(i, j, m + bw - 1, bw)]);
		fprintf(stderr, "\n");
	}
};
#endif
#endif //SW_AVX_H