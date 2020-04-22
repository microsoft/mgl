#include "pairhmm_common.h"

void computeHaplotypeSimilarities(std::vector<hapinfo> & hap_array, size_t &COLS_MIN, size_t &COLS_MAX)
{
	size_t numHaplotypes = hap_array.size();

	if (numHaplotypes == 0)
	{
		return;
	}

	// Compute haplotype similarities
	hap_array[0].position = 0;
	hap_array[0].score = 0.0;

	COLS_MIN = COLS_MAX = hap_array[0].haplen;

	for (size_t hap_idx = 1; hap_idx < numHaplotypes; ++hap_idx)
	{
		size_t pos = 0;

		if (hap_array[hap_idx - 1].haplen >= 8)
		{
			size_t commonLen = min(hap_array[hap_idx - 1].haplen, hap_array[hap_idx].haplen);

			for (; pos < commonLen; pos++)
			{
				if (hap_array[hap_idx - 1].hap[pos] != hap_array[hap_idx].hap[pos])
					break;
			}

			if (pos % 2 == 1)
			{
				pos -= 1;
			}

			if (pos < hap_array[hap_idx - 1].position)
			{
				pos = 0;
			}
		}

		//int maxpos = 2;
		//if (pos > maxpos)
		//{
		//	hap_array[hap_idx].position = maxpos;
		//}

		hap_array[hap_idx].position = pos;
		//hap_array[hap_idx].position = 0;
		hap_array[hap_idx].score = 0.0;

		if (COLS_MIN > hap_array[hap_idx].haplen)
		{
			COLS_MIN = hap_array[hap_idx].haplen;
		}

		if (COLS_MAX < hap_array[hap_idx].haplen)
		{
			COLS_MAX = hap_array[hap_idx].haplen;
		}
	}
}

void DebugDump(
	float** M, float ** X, float ** Y,
	size_t ROWS, size_t COLS)
{
	FILE* fTable;
	fopen_s(&fTable, "pairHmm.csv", "w");

	fprintf(fTable, ", ");
	for (size_t cc = 0; cc < COLS; cc++)
	{
		fprintf(fTable, "%zd, ", cc);
	}
	fprintf(fTable, "\n\n");

	for (size_t rr = 0; rr < ROWS; rr++)
	{
		fprintf(fTable, "%zd, ", rr);
		for (size_t cc = 0; cc < COLS; cc++)
		{
			fprintf(fTable, "%f, ", M[rr][cc]);
		}
		fprintf(fTable, "\n");

		fprintf(fTable, ", ");
		for (size_t cc = 0; cc < COLS; cc++)
		{
			fprintf(fTable, "%f, ", X[rr][cc]);
		}
		fprintf(fTable, "\n");

		fprintf(fTable, ", ");
		for (size_t cc = 0; cc < COLS; cc++)
		{
			fprintf(fTable, "%f, ", Y[rr][cc]);
		}
		fprintf(fTable, "\n\n");
	}

	fflush(fTable);
	fclose(fTable);
}

void DebugDump(
	double** M, double ** X, double ** Y,
	size_t ROWS, size_t COLS)
{
	FILE* fTable;
	fopen_s(&fTable, "pairHmm.csv", "w");

	fprintf(fTable, ", ");
	for (size_t cc = 0; cc < COLS; cc++)
	{
		fprintf(fTable, "%zd, ", cc);
	}
	fprintf(fTable, "\n\n");

	for (size_t rr = 0; rr < ROWS; rr++)
	{
		fprintf(fTable, "%zd, ", rr);
		for (size_t cc = 0; cc < COLS; cc++)
		{
			fprintf(fTable, "%e, ", M[rr][cc]);
		}
		fprintf(fTable, "\n");

		fprintf(fTable, ", ");
		for (size_t cc = 0; cc < COLS; cc++)
		{
			fprintf(fTable, "%e, ", X[rr][cc]);
		}
		fprintf(fTable, "\n");

		fprintf(fTable, ", ");
		for (size_t cc = 0; cc < COLS; cc++)
		{
			fprintf(fTable, "%e, ", Y[rr][cc]);
		}
		fprintf(fTable, "\n\n");
	}

	fflush(fTable);
	fclose(fTable);
}
