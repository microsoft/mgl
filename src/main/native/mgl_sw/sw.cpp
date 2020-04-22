#include"sw_scalar.h"

using namespace std;

void calculateMatrix(const char *ts, int tl, const char *qs, int ql, int * bcktrack, swParameters parameters, int overhangStrategy, ScoreMax *ez)
{
	int n = tl + 1, m = ql + 1, i, j;
	int inf = INT_MIN / 2,  *f, *e, *gap_v, *gap_h, *sc;

	sc = (int*)malloc(m * sizeof(int));
	memset(sc, 0, m * sizeof(int));
	e = (int*)malloc(m * sizeof(int));
	gap_v = (int*)malloc(m * sizeof(int));
	for (i = 0; i < m; i++)
	{
		e[i] = -parameters.g_open;
		gap_v[i] = 1;
	}

	f = (int*)malloc(n * sizeof(int));
	gap_h = (int*)malloc(n * sizeof(int));
	for (i = 0; i < n; i++)
	{
		f[i] = -parameters.g_open;
		gap_h[i] = 1;
	}

	// initialize first row and first column, modify initial conditions
	if ((overhangStrategy & SW_OS_INDEL) | (overhangStrategy & SW_OS_LEAD_ID))
	{
		for (i = 1; i < m; i++)
		{
			sc[i] = -parameters.g_open - (i - 1) * parameters.g_ext;
			e[i] += -parameters.g_open - (i - 1) * parameters.g_ext;
		}
		for (i = 1; i < n; i++)
		{
			f[i] += -parameters.g_open - (i - 1) * parameters.g_ext;
		}
	}

	int a, b, step_diag, step_right, step_down, gap_down, gap_right;
	int sc_prev, sc_cur;
	// calculate score and backtrack matrices
	for (i = 1; i < n; i++)
	{
		sc_prev = 0;
		if ((overhangStrategy & SW_OS_INDEL) | (overhangStrategy & SW_OS_LEAD_ID))
			sc_prev = -parameters.g_open - (i - 1) * parameters.g_ext;

		a = ts[i - 1];
		for (j = 1; j < m; j++)
		{
			b = qs[j - 1];
			step_diag = sc[j - 1] + (a == b ? parameters.sc_match : parameters.sc_mismatch);
			step_down = e[j]; gap_down = gap_v[j];
			step_right = f[i]; gap_right = gap_h[i];

			//priority here will be step diagonal, step right, step down
			if ((step_diag >= step_down) && (step_diag >= step_right)) {
				sc_cur =  step_diag;
				bcktrack[i * m + j] = 0;
			}
			else if (step_right >= step_down) { //moving right is the highest
				sc_cur = step_right;
				bcktrack[i * m + j] = -gap_right; 
			}
			else {
				sc_cur = step_down;
				bcktrack[i * m + j] = gap_down; 
			}
			// update e and f for the next step
			if (sc_cur - parameters.g_open > e[j] - parameters.g_ext)
			{
				e[j] = sc_cur - parameters.g_open;
				gap_v[j] = 1;
			}
			else
			{
				e[j] -= parameters.g_ext;
				gap_v[j]++;
			}

			if (sc_cur - parameters.g_open > f[i] - parameters.g_ext)
			{
				f[i] = sc_cur - parameters.g_open;
				gap_h[i] = 1;
			}
			else 
			{
				f[i] -= parameters.g_ext;
				gap_h[i]++;
			}
			// save score value for the next step
			sc[j - 1] = sc_prev;
			sc_prev = sc_cur;
		}
		sc[j - 1] = sc_prev;
		// find max score in the last column
		if (sc_cur >= ez->mqe)
		{
			ez->mqe_t = i;
			ez->mqe = sc_cur;
		}
	}

// look for the largest score on the rightmost column. we use >= combined with the traversal direction
// to ensure that if two scores are equal, the one closer to diagonal gets picked
//!!!!!
// Method above doesn't guarantee that "the one closer to diagonal gets picked" TO DO - contact Broad to get clarifications.
//!!!!!
//Note: this is not technically smith-waterman, as by only looking for max values on the right we are
//excluding high scoring local alignments

	// find max score in the last row
	ez->max = ez->mqe; ez->max_t = ez->mqe_t; ez->max_q = ql;
	for (j = 1; j < m; j++)
	{
		sc_cur = sc[j];
		if (sc_cur > ez->max || (sc_cur == ez->max && abs(tl - j) < abs(ez->max_t - ez->max_q)))
		{
			ez->max_t = tl;
			ez->max_q = j;
			ez->max = sc_cur;
			ez->seg_length = ql - j;
		}
	}

#ifdef DEBUG_PRINT
	fprintf(stderr, "Backtrack\n");
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
			fprintf(stderr, "%d\t", bcktrack[i * m + j]);
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
#endif


	free(sc);
	free(e);
	free(f);
	free(gap_v);
	free(gap_h);
};


int calculateCigar(int *bcktrack, int n, int m, int overhangStrategy, ScoreMax *ez, string * cigar)
{
	int I = 0, J = 0;
	int refLength = n - 1, altLength = m - 1, segment_length = 0;

	// define start point for bactracking
	if (overhangStrategy == SW_OS_INDEL)
	{
		I = refLength;
		J = altLength;
	}
	else if (overhangStrategy != SW_OS_LEAD_ID)
	{
		I = ez->max_t;
		J = ez->max_q;
		segment_length = ez->seg_length;
	}
	else 
	{
		I = ez->mqe_t;
		J = altLength;
	}

	list<CigarElement>  result;
	if (segment_length > 0 && overhangStrategy == SW_OS_SOFTCLIP) {
		result.push_front(CigarElement(STATE_CLIP, segment_length));
		segment_length = 0;
	}

	// we will be placing all insertions and deletions into sequence b, so the states are named w/regard
	// to that sequence
	char state = STATE_MATCH, nextState;
	int btr, step_length;
	do {
		step_length = 1;
		btr = bcktrack[I * m + J];
		if (btr > 0)
		{
			nextState = STATE_DEL;
			step_length = btr;
		}
		else if (btr < 0)
		{
			nextState = STATE_INS;
			step_length = -btr;
		}
		else nextState = STATE_MATCH;

		// move to next best location in the sw matrix:
		switch (nextState)
		{
		case STATE_MATCH:  I--; J--; break; // move back along the diag in the sw matrix
		case STATE_INS: J -= step_length; break; // move left
		case STATE_DEL:  I -= step_length; break; // move up
		}

		// now let's see if the state actually changed:
		if (nextState == state) segment_length += step_length;
		else {
			// state changed, lets emit previous segment, whatever it was (Insertion Deletion, or (Mis)Match).
			result.push_front(CigarElement(state, segment_length));
			segment_length = step_length;
			state = nextState;
		}
		// next condition is equivalent to  while ( sw[p1][p2] != 0 ) (with modified p1 and/or p2:
	} while (I > 0 && J > 0);

	// post-process the last segment we are still keeping;
	// NOTE: if reads "overhangs" the ref on the left (i.e. if p2>0) we are counting
	// those extra bases sticking out of the ref into the first cigar element if DO_SOFTCLIP is false;
	// otherwise they will be softclipped. For instance,
	// if read length is 5 and alignment starts at offset -2 (i.e. read starts before the ref, and only
	// last 3 bases of the read overlap with/align to the ref), the cigar will be still 5M if
	// DO_SOFTCLIP is false or 2S3M if DO_SOFTCLIP is true.
	// The consumers need to check for the alignment offset and deal with it properly.
	int alignment_offset;
	if (overhangStrategy == SW_OS_SOFTCLIP) {
		result.push_front(CigarElement(state, segment_length));
		if (J > 0) result.push_front(CigarElement(STATE_CLIP, J));
		alignment_offset = I;
	}
	else if (overhangStrategy == SW_OS_IGNORE) {
		result.push_front(CigarElement(state, segment_length + J));
		alignment_offset = I - J;
	}
	else {  // overhangStrategy == OverhangStrategy.INDEL || overhangStrategy == OverhangStrategy.LEADING_INDEL

	 // take care of the actual alignment
		result.push_front(CigarElement(state, segment_length));

		// take care of overhangs at the beginning of the alignment
		if (I > 0) {
			result.push_front(CigarElement(STATE_DEL, I));
		}
		else if (J > 0) {
			result.push_front(CigarElement(STATE_INS, J));
		}

		alignment_offset = 0;
	}

	// convert list of elements into cigar string
	for (list<CigarElement>::iterator it = result.begin(); it != result.end(); ++it)
		if (it->length > 0) *cigar = *cigar + to_string(it->length) + it->state;

	return alignment_offset;
}


int align_scalar(const char *tseq, int target_length, const char *qseq, int query_length, swParameters parameters, int strategy, string * result_cigar)
{
	ScoreMax ez;

	int n = target_length + 1, m = query_length + 1;
	int * bcktrack;
	bcktrack = (int*)malloc(n * m * sizeof(int));
	memset(bcktrack, 0, n * m * sizeof(int));
	   
	calculateMatrix(tseq, target_length, qseq, query_length, bcktrack, parameters, strategy, &ez);
	int offset = calculateCigar(bcktrack, n, m, strategy, &ez, result_cigar);

	free(bcktrack);
	return offset;
}

