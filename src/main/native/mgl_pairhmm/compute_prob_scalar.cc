#include "compute_prob_scalar.h"

using namespace std;

template<class NUMBER>
void compute_full_prob_core_1X(
	size_t startRow, size_t startCol, size_t iterations,
	char * rs, char * hap,
	NUMBER * Distm,
#ifdef _UNIT_TEST
	NUMBER * DistmDiag,
#endif //_UNIT_TEST
	NUMBER * pMM, NUMBER * pGapM, NUMBER * pMX, NUMBER * pMY, NUMBER * pZZ,
	NUMBER * MDiag0, NUMBER * XDiag0, NUMBER * YDiag0,
	NUMBER * MDiag1, NUMBER * XDiag1, NUMBER * YDiag1)
{
	const NUMBER threeOver = NUMBER(1.0) / NUMBER(3.0);

	for (size_t r = startRow + iterations, c = startCol - iterations; iterations > 0; iterations--, r--, c++)
	{
		assert(r > 0);
		assert(c > 0);

		char _rs = rs[r - 1];
		char _hap = hap[c - 1];

		bool bMatch = ((_rs == _hap) | (_rs == 'N') | (_hap == 'N'));

		NUMBER distm = Distm[r];

		if (bMatch)
			distm = NUMBER(1.0) - distm;
		else
			distm = distm * threeOver;
#ifdef _UNIT_TEST
		DistmDiag[r] = distm;
#endif //_UNIT_TEST

		MDiag0[r] = distm * (MDiag0[r - 1] * pMM[r] + (XDiag0[r - 1] + YDiag0[r - 1]) * pGapM[r]);

		YDiag0[r] = MDiag1[r] * pMY[r] + YDiag1[r] * pZZ[r];

		XDiag0[r] = MDiag1[r - 1] * pMX[r] + XDiag1[r - 1] * pZZ[r];
	}
}

template<class NUMBER>
void compute_prob_scalar(readinfo &read, vector<hapinfo> &hap_array)
{
	Context<NUMBER> ctx;

	const NUMBER threeOver = ctx._(1.0) / ctx._(3.0);

	size_t ROWS = read.rslen + 1;

	NUMBER* pAll = new NUMBER[ROWS * 5];

	NUMBER* pMM = pAll;
	NUMBER* pGapM = pMM + ROWS;
	NUMBER* pMX = pGapM + ROWS;
	NUMBER* pMY = pMX + ROWS;
	NUMBER* pZZ = pMY + ROWS;

	NUMBER* Distm = new NUMBER[ROWS];

	pMM[0] = ctx._(0.0);
	pGapM[0] = ctx._(0.0);
	pMX[0] = ctx._(0.0);
	pMY[0] = ctx._(0.0);
	pZZ[0] = ctx._(0.0);

	Distm[0] = ctx._(0.0);

	for (size_t r = 1; r < ROWS; r++)
	{
		int _i = read.i[r - 1] & 127;
		int _d = read.d[r - 1] & 127;
		int _c = read.c[r - 1] & 127;
		pMM[r] = ctx.set_mm_prob(_i, _d);
		pGapM[r] = ctx._(1.0) - ctx.ph2pr[_c];
		pMX[r] = ctx.ph2pr[_i];
		pMY[r] = ctx.ph2pr[_d];
		pZZ[r] = ctx.ph2pr[_c];

		Distm[r] = ctx.ph2pr[read.q[r - 1] & 127];
	}

	NUMBER* MDiag0 = new NUMBER[ROWS];
	NUMBER* MDiag1 = new NUMBER[ROWS];
	NUMBER* XDiag0 = new NUMBER[ROWS];
	NUMBER* XDiag1 = new NUMBER[ROWS];
	NUMBER* YDiag0 = new NUMBER[ROWS];
	NUMBER* YDiag1 = new NUMBER[ROWS];

#ifdef _UNIT_TEST
	NUMBER* DistmDiag = new NUMBER[ROWS];
#endif // _UNIT_TEST

	size_t numHaplotypes = hap_array.size();

	for (size_t hap_idx = 0; hap_idx < numHaplotypes; ++hap_idx)
	{
		size_t COLS = hap_array[hap_idx].haplen + 1;
		size_t DIAGS = ROWS + COLS - 1;

		NUMBER yInitial = ctx.INITIAL_CONSTANT / hap_array[hap_idx].haplen;

#ifdef _UNIT_TEST
		NUMBER** M = DebugCompute(ctx, read, hap_array[hap_idx], yInitial);
		NUMBER** X = M + ROWS;
		NUMBER** Y = X + ROWS;

		NUMBER resultDebug = NUMBER(0.0);

		for (size_t c = 0; c < COLS; c++)
		{
			resultDebug += M[ROWS - 1][c] + X[ROWS - 1][c];
		}

#ifdef _UNIT_TEST_DUMP
		DebugDump(M, X, Y, ROWS, COLS);
#endif // _UNIT_TEST_DUMP

#endif // _UNIT_TEST

		MDiag0[0] = ctx._(0.0);
		XDiag0[0] = ctx._(0.0);
		YDiag0[0] = yInitial;

		MDiag1[0] = ctx._(0.0);
		XDiag1[0] = ctx._(0.0);
		YDiag1[0] = yInitial;

		MDiag1[1] = ctx._(0.0);
		XDiag1[1] = ctx._(0.0);
		YDiag1[1] = ctx._(0.0);

		NUMBER result = NUMBER(0.0);

		size_t i = 2;
		size_t startRow = 0;
		size_t iterations = 0;

		if (COLS >= ROWS)
		{
			for (; i < ROWS; i++)
			{
				MDiag0[i] = ctx._(0.0);
				XDiag0[i] = ctx._(0.0);
				YDiag0[i] = ctx._(0.0);

				iterations++;

				assert(iterations < ROWS);
				assert(iterations < i);

				compute_full_prob_core_1X(
					0, i, iterations,
					read.rs, hap_array[hap_idx].hap,
					Distm,
#ifdef _UNIT_TEST
					DistmDiag,
#endif //_UNIT_TEST
					pMM, pGapM, pMX, pMY, pZZ,
					MDiag0, XDiag0, YDiag0,
					MDiag1, XDiag1, YDiag1);

				MDiag0[0] = ctx._(0.0);
				XDiag0[0] = ctx._(0.0);
				YDiag0[0] = ctx.INITIAL_CONSTANT / hap_array[hap_idx].haplen;

#ifdef _UNIT_TEST
				DebugVerify(0, i, iterations, MDiag0, XDiag0, YDiag0, M, X, Y);
#endif //_UNIT_TEST

				NUMBER *MTemp = MDiag0, *XTemp = XDiag0, *YTemp = YDiag0;
				MDiag0 = MDiag1; XDiag0 = XDiag1; YDiag0 = YDiag1;
				MDiag1 = MTemp; XDiag1 = XTemp; YDiag1 = YTemp;
			}

			iterations++;

			for (; i < COLS; i++)
			{
				assert(iterations < ROWS);
				assert(iterations < i);

				compute_full_prob_core_1X(
					0, i, iterations,
					read.rs, hap_array[hap_idx].hap,
					Distm,
#ifdef _UNIT_TEST
					DistmDiag,
#endif //_UNIT_TEST
					pMM, pGapM, pMX, pMY, pZZ,
					MDiag0, XDiag0, YDiag0,
					MDiag1, XDiag1, YDiag1);

#ifdef _UNIT_TEST
				DebugVerify(0, i, iterations, MDiag0, XDiag0, YDiag0, M, X, Y);
#endif //_UNIT_TEST

				MDiag0[0] = ctx._(0.0);
				XDiag0[0] = ctx._(0.0);
				YDiag0[0] = ctx.INITIAL_CONSTANT / hap_array[hap_idx].haplen;

				NUMBER *MTemp = MDiag0, *XTemp = XDiag0, *YTemp = YDiag0;
				MDiag0 = MDiag1; XDiag0 = XDiag1; YDiag0 = YDiag1;
				MDiag1 = MTemp; XDiag1 = XTemp; YDiag1 = YTemp;

				result += MDiag1[ROWS - 1] + XDiag1[ROWS - 1];
			}
		}
		else
		{
			for (; i < COLS; i++)
			{
				MDiag0[i] = ctx._(0.0);
				XDiag0[i] = ctx._(0.0);
				YDiag0[i] = ctx._(0.0);

				iterations++;

				assert(iterations < ROWS);
				assert(iterations < i);

				compute_full_prob_core_1X(
					0, i, iterations,
					read.rs, hap_array[hap_idx].hap,
					Distm,
#ifdef _UNIT_TEST
					DistmDiag,
#endif //_UNIT_TEST
					pMM, pGapM, pMX, pMY, pZZ,
					MDiag0, XDiag0, YDiag0,
					MDiag1, XDiag1, YDiag1);

				MDiag0[0] = ctx._(0.0);
				XDiag0[0] = ctx._(0.0);
				YDiag0[0] = ctx.INITIAL_CONSTANT / hap_array[hap_idx].haplen;

#ifdef _UNIT_TEST
				DebugVerify(0, i, iterations, MDiag0, XDiag0, YDiag0, M, X, Y);
#endif //_UNIT_TEST

				NUMBER *MTemp = MDiag0, *XTemp = XDiag0, *YTemp = YDiag0;
				MDiag0 = MDiag1; XDiag0 = XDiag1; YDiag0 = YDiag1;
				MDiag1 = MTemp; XDiag1 = XTemp; YDiag1 = YTemp;
			}

			iterations++;

			for (; i < ROWS; i++)
			{
				MDiag0[i] = ctx._(0.0);
				XDiag0[i] = ctx._(0.0);
				YDiag0[i] = ctx._(0.0);

				assert(startRow + iterations < ROWS);
				assert(iterations < COLS);

				compute_full_prob_core_1X(
					startRow, COLS, iterations,
					read.rs, hap_array[hap_idx].hap,
					Distm,
#ifdef _UNIT_TEST
					DistmDiag,
#endif //_UNIT_TEST
					pMM, pGapM, pMX, pMY, pZZ,
					MDiag0, XDiag0, YDiag0,
					MDiag1, XDiag1, YDiag1);

#ifdef _UNIT_TEST
				DebugVerify(startRow, COLS, iterations, MDiag0, XDiag0, YDiag0, M, X, Y);
#endif //_UNIT_TEST

				MDiag0[0] = ctx._(0.0);
				XDiag0[0] = ctx._(0.0);
				YDiag0[0] = ctx.INITIAL_CONSTANT / hap_array[hap_idx].haplen;

				NUMBER *MTemp = MDiag0, *XTemp = XDiag0, *YTemp = YDiag0;
				MDiag0 = MDiag1; XDiag0 = XDiag1; YDiag0 = YDiag1;
				MDiag1 = MTemp; XDiag1 = XTemp; YDiag1 = YTemp;

				startRow++;
			}
		}

		for (; i < DIAGS; i++)
		{
			assert(startRow + iterations < ROWS);
			assert(iterations < COLS);

			compute_full_prob_core_1X(
				startRow, COLS, iterations,
				read.rs, hap_array[hap_idx].hap,
				Distm,
#ifdef _UNIT_TEST
				DistmDiag,
#endif //_UNIT_TEST
				pMM, pGapM, pMX, pMY, pZZ,
				MDiag0, XDiag0, YDiag0,
				MDiag1, XDiag1, YDiag1);

#ifdef _UNIT_TEST
			DebugVerify(startRow, COLS, iterations, MDiag0, XDiag0, YDiag0, M, X, Y);
#endif //_UNIT_TEST

			NUMBER *MTemp = MDiag0, *XTemp = XDiag0, *YTemp = YDiag0;
			MDiag0 = MDiag1; XDiag0 = XDiag1; YDiag0 = YDiag1;
			MDiag1 = MTemp; XDiag1 = XTemp; YDiag1 = YTemp;

			result += MDiag1[ROWS - 1] + XDiag1[ROWS - 1];

			startRow++;
			iterations--;
		}

		hap_array[hap_idx].score = double(result);

#ifdef _UNIT_TEST
		DebugAssertClose(result, resultDebug);

		delete[] M[0];
		delete[] M;
#endif //_UNIT_TEST
	}

	delete[] MDiag0;
	delete[] MDiag1;
	delete[] XDiag0;
	delete[] XDiag1;
	delete[] YDiag0;
	delete[] YDiag1;
#ifdef _UNIT_TEST
	delete[] DistmDiag;
#endif //_UNIT_TEST

	delete[] Distm;

	delete[] pAll;
}

void compute_prob_scalarf(readinfo &read, vector<hapinfo> &hap_array)
{
	compute_prob_scalar<float>(read, hap_array);
}

void compute_prob_scalard(readinfo &read, vector<hapinfo> &hap_array)
{
	compute_prob_scalar<double>(read, hap_array);
}

