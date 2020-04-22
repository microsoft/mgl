#include"sw_avx.h"

using namespace std;


int align_avx(const char *tseq, int target_length, const char *qseq, int query_length, swParameters parameters, int overhangStrategy, string * result_cigar)
{
	int i;
	ScoreMax ez;
	int nrow = target_length + 1, ncol = query_length + 1;
	int default_bw = 8, padding = default_bw;
	size_t alignment = 32;
	int *reversed_query, *extended_target;
	int ext_query_length = query_length + 2 * padding, ext_target_length; 

	// extend and reverse query to use in score and backtrack matrix computation
	reversed_query = (int*)_aligned_malloc(ext_query_length * sizeof(int), alignment);
	memset(reversed_query, 0, ext_query_length * sizeof(int));
	for (int i = 0; i < query_length; i++)
		reversed_query[padding + query_length - 1 - i] = (int)qseq[i];

	// extend target to have number of rows divisible by default_bw
	int target_padding;
	target_padding = (target_length % default_bw == 0) ? 0 : (default_bw - target_length % default_bw);
	ext_target_length = target_length + target_padding;
	extended_target = (int*)_aligned_malloc(ext_target_length * sizeof(int), alignment);
	memset(extended_target, 0, ext_target_length * sizeof(int));
	for (int i = 0; i < target_length; i++)
		extended_target[i] = (int)tseq[i];

	// backtrack matrix
	int * bcktrack;
	bcktrack = (int*)_aligned_malloc((query_length + padding - 1) * (target_length + target_padding) * sizeof(int), alignment);
	memset(bcktrack, 0, (query_length + padding - 1) * (target_length + target_padding) * sizeof(int));

	// arrays to save score, vertical step and vertical gap values for the next score band
	int *score, *step, *gap;
	score = (int*)_aligned_malloc( (ncol + padding) * sizeof(int), alignment);
	memset(score, 0, (ncol + padding) * sizeof(int));
	step = (int*)_aligned_malloc((ncol + padding) * sizeof(int), alignment);
	memset(step, 0, (ncol + padding) * sizeof(int));
	gap = (int*)_aligned_malloc((query_length + 2 * padding) * sizeof(int), alignment);
	memset(gap, 0, (query_length + 2 * padding) * sizeof(int));

	// initial conditions
	for (i = 0; i < ncol; i++)
		step[i] = -parameters.g_open;
	for (i = 0; i < query_length + 2 * padding; i++)
 		gap[i] = 1;


	// Modify initial condition, non-zero initials for Indel and Leading_indel overhang strategy
	if ((overhangStrategy & SW_OS_INDEL) | (overhangStrategy & SW_OS_LEAD_ID))
	{
		for (i = 1; i < ncol; i++)
		{
			score[i] = -parameters.g_open - (i - 1) * parameters.g_ext;
			step[i] += -parameters.g_open - (i - 1) * parameters.g_ext;
		}
	}

#ifdef DEBUG_PRINT	
	fprintf(stderr, "Score\n");
	print_array(stderr, score, ncol);
	fprintf(stderr, "Step_down\n");
	print_array(stderr, step, ncol);
	fprintf(stderr, "Gap_down\n");
	print_array(stderr, gap, query_length + 2 * padding);
#endif

	int band_count = 0, total_bands = (target_length % default_bw) ? (target_length / default_bw + 1) : (target_length / default_bw), rows_left = target_length;
	while (band_count < total_bands)
	{
		int nrow_new = (rows_left >= default_bw) ? default_bw : rows_left; //number of rows in current band
		rows_left -= nrow_new;

		// fill backtrack matrix
		calculateMatrix_avx(extended_target, target_length, reversed_query, query_length, bcktrack, band_count, default_bw, nrow_new, score, step, gap, parameters, overhangStrategy, &ez);
		band_count++;
	}

	// Update ez using score array as last row
	ez.max = ez.mqe; ez.max_t = ez.mqe_t; ez.max_q = query_length;
	for (i = 1; i < ncol; i++)
	{
		int sc_cur = score[i];
		if (sc_cur > ez.max || (sc_cur == ez.max && abs(target_length - i) < abs(ez.max_t - ez.max_q)))
		{
			ez.max_t = target_length;
			ez.max_q = i;
			ez.max = sc_cur;
			ez.seg_length = query_length - i;
		}
	}
	
	
	int offset = calculateCigar_avx(bcktrack, nrow, ncol, default_bw, overhangStrategy, &ez, result_cigar);
	
	_aligned_free(reversed_query);
	_aligned_free(extended_target);
	_aligned_free(score);
	_aligned_free(step);
	_aligned_free(gap);
	_aligned_free(bcktrack);

	return offset;
	
}

void calculateMatrix_avx(int *target, int target_length, int *query, int query_length, int *bcktrack, int band_count, int default_bw, int actual_bw, int *score, int *step, int *gap, swParameters parameters, int overhangStrategy, ScoreMax *ez)
{
	__m256i _S0, _S1, _E, _F, _gap_v, _gap_h = _mm256_set1_epi32(-1), _direction = _mm256_setzero_si256();
	__m256i _gapo = _mm256_set1_epi32(parameters.g_open), _gape = _mm256_set1_epi32(parameters.g_ext);
	__m256i _s_col = _mm256_setzero_si256(), _f_col = _mm256_set1_epi32(-parameters.g_open);

	// modify initial condition, non-zero initials for Indel and Leading_indel overhang strategy
	if ((overhangStrategy & SW_OS_INDEL) | (overhangStrategy & SW_OS_LEAD_ID))
	{
		int s_col[8];
		for (int i = 0; i < 8; i++)
			s_col[i] = -parameters.g_open - (i + default_bw * band_count) * parameters.g_ext;
		_s_col = _mm256_load_si256((__m256i*)&s_col);
		_f_col = _mm256_add_epi32(_f_col, _s_col);
	}

	int match = parameters.sc_match, mismatch = parameters.sc_mismatch, int_min = INT_MIN / 2, padding = default_bw;
	int ncol = query_length + 1;

	// initialize first diagonals
	_S1 = _mm256_set1_epi32(int_min);
	_S0 = _mm256_set_epi32(int_min, int_min, int_min, int_min, int_min, int_min, int_min, score[0]);
	_E = _mm256_set_epi32(int_min, int_min, int_min, int_min, int_min, int_min, int_min, step[0]);
	_F = _mm256_set1_epi32(int_min);

	__m256i _tmp, _query, _target = _mm256_load_si256((__m256i*)(target + default_bw * band_count));

	// masks for the first triangle
	__m256i _scMask, _bcktrMask, _hMask = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1);

	// fill first triangle
	int i = 0;
	// i = 0
	// set score initial conditions
	_S1 = _mm256_blendv_epi8(_S1, _s_col, _hMask);
	// calculate E and F for next step
	_tmp = _mm256_sub_epi32(_S1, _gapo);
	_E = _mm256_sub_epi32(_E, _gape);
	_E = _mm256_max_epi32(_tmp, _E);
	_F = _mm256_sub_epi32(_F, _gape);
	_F = _mm256_max_epi32(_tmp, _F);
	// set F initial conditions
	_F = _mm256_blendv_epi8(_F, _f_col, _hMask);
	// shift _S, _E, _hMask
	_S1 = _mm256_shift_left_si256<4>(_S1, _mm256_set1_epi32(score[i + 1]));
	_E = _mm256_shift_left_si256<4>(_E, _mm256_set1_epi32(step[i + 1]));
	_hMask = _mm256_shift_left_si256<4>(_hMask);
	_tmp = _S0; _S0 = _S1; _S1 = _tmp;

	for (i = 1; i < actual_bw; i++)
	{
		_gap_v = _mm256_load_si256((__m256i*)(gap + padding + query_length - 1 - (i - 1)));
		_query = _mm256_load_si256((__m256i*)(query + padding + query_length - 1 - (i - 1)));
		_scMask = _mm256_cmpeq_epi32(_query, _target);
		_tmp = _mm256_add_epi32(_mm256_and_si256(_scMask, _mm256_set1_epi32(match)), _mm256_andnot_si256(_scMask, _mm256_set1_epi32(mismatch)));
		_S1 = _mm256_add_epi32(_S1, _tmp);

		_bcktrMask = _mm256_cmpgt_epi32(_F, _S1);
		_direction = _mm256_and_si256(_bcktrMask, _gap_h);
		_S1 = _mm256_max_epi32(_S1, _F);

		_bcktrMask = _mm256_cmpgt_epi32(_E, _S1);
		_direction = _mm256_blendv_epi8(_direction, _gap_v, _bcktrMask);
		_mm256_store_si256((__m256i*)(bcktrack + (query_length + padding - 1) * default_bw * band_count + (i - 1) * default_bw), _direction);
		_S1 = _mm256_max_epi32(_S1, _E);

		// set score initial conditions
		_S1 = _mm256_blendv_epi8(_S1, _s_col, _hMask);

		_tmp = _mm256_sub_epi32(_S1, _gapo);
		_E = _mm256_sub_epi32(_E, _gape);
		_bcktrMask = _mm256_cmpgt_epi32(_tmp, _E);
		_gap_v = _mm256_blendv_epi8(_mm256_add_epi32(_gap_v, _mm256_set1_epi32(1)), _mm256_set1_epi32(1), _bcktrMask);
		_mm256_store_si256((__m256i*)(gap + padding + query_length - 1 - (i - 1)), _gap_v);
		_E = _mm256_max_epi32(_tmp, _E);

		_F = _mm256_sub_epi32(_F, _gape);
		_bcktrMask = _mm256_cmpgt_epi32(_tmp, _F);
		_gap_h = _mm256_blendv_epi8(_mm256_sub_epi32(_gap_h, _mm256_set1_epi32(1)), _mm256_set1_epi32(-1), _bcktrMask);
		_F = _mm256_max_epi32(_tmp, _F);
		_F = _mm256_blendv_epi8(_F, _f_col, _hMask);
		_gap_h = _mm256_blendv_epi8(_gap_h, _mm256_set1_epi32(-1), _hMask);

		//save border values for the next band
		if (i + 1 - actual_bw >= 0)
		{
			score[i + 1 - actual_bw] = _mm256_get_epi32(_S1, actual_bw - 1); //_mm256_extract_epi32(_S1, 7); 
			step[i + 1 - actual_bw] = _mm256_get_epi32(_E, actual_bw - 1); 
		}

		// shift _S, _E, _hMask
		_S1 = _mm256_shift_left_si256<4>(_S1, _mm256_set1_epi32(score[i + 1]));
		_E = _mm256_shift_left_si256<4>(_E, _mm256_set1_epi32(step[i + 1]));
		_hMask = _mm256_shift_left_si256<4>(_hMask);

		_tmp = _S0; _S0 = _S1; _S1 = _tmp;
	}


	// fill middle part
	for (; i < ncol; i++)
	{
		_gap_v = _mm256_load_si256((__m256i*)(gap + padding + query_length - 1 - (i - 1)));
		_query = _mm256_load_si256((__m256i*)(query + padding + query_length - 1 - (i - 1)));
		_scMask = _mm256_cmpeq_epi32(_query, _target);
		_tmp = _mm256_add_epi32(_mm256_and_si256(_scMask, _mm256_set1_epi32(match)), _mm256_andnot_si256(_scMask, _mm256_set1_epi32(mismatch)));
		_S1 = _mm256_add_epi32(_S1, _tmp);

		_bcktrMask = _mm256_cmpgt_epi32(_F, _S1);
		_direction = _mm256_and_si256(_bcktrMask, _gap_h);
		_S1 = _mm256_max_epi32(_S1, _F);

		_bcktrMask = _mm256_cmpgt_epi32(_E, _S1);
		_direction = _mm256_blendv_epi8(_direction, _gap_v, _bcktrMask);
		_mm256_store_si256((__m256i*)(bcktrack + (query_length + padding - 1) * default_bw * band_count + (i - 1) * default_bw), _direction);
		_S1 = _mm256_max_epi32(_S1, _E);

		_tmp = _mm256_sub_epi32(_S1, _gapo);
		_E = _mm256_sub_epi32(_E, _gape);
		_bcktrMask = _mm256_cmpgt_epi32(_tmp, _E);
		_gap_v = _mm256_blendv_epi8(_mm256_add_epi32(_gap_v, _mm256_set1_epi32(1)), _mm256_set1_epi32(1), _bcktrMask);
		_mm256_store_si256((__m256i*)(gap + padding + query_length - 1 - (i - 1)), _gap_v);
		_E = _mm256_max_epi32(_tmp, _E);

		_F = _mm256_sub_epi32(_F, _gape);
		_bcktrMask = _mm256_cmpgt_epi32(_tmp, _F);
		_gap_h = _mm256_blendv_epi8(_mm256_sub_epi32(_gap_h, _mm256_set1_epi32(1)), _mm256_set1_epi32(-1), _bcktrMask);
		_F = _mm256_max_epi32(_tmp, _F);

		//save border values for the next band
		score[i + 1 - actual_bw] = _mm256_get_epi32(_S1, actual_bw - 1); 
		step[i + 1 - actual_bw] = _mm256_get_epi32(_E, actual_bw - 1);

		// shift S, E
		_S1 = _mm256_shift_left_si256<4>(_S1, _mm256_set1_epi32(score[i + 1]));
		_E = _mm256_shift_left_si256<4>(_E, _mm256_set1_epi32(step[i + 1]));

		_tmp = _S0; _S0 = _S1; _S1 = _tmp;
	}

	// save last column of score matrix, we'll need last column to update ez
	// save first element from last column, ugly initialization - replace later?
	__m256i _last_col = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, _mm256_extract_epi32(_S0, 1)); 

	_hMask = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0);
	//fill last triangle
	for ( ; i < ncol + actual_bw - 1; i++)
	{
		_gap_v = _mm256_load_si256((__m256i*)(gap + padding + query_length - 1 - (i - 1)));
		_query = _mm256_load_si256((__m256i*)(query + padding + query_length - 1 - (i - 1)));
		_scMask = _mm256_cmpeq_epi32(_query, _target);
		_tmp = _mm256_add_epi32(_mm256_and_si256(_scMask, _mm256_set1_epi32(match)), _mm256_andnot_si256(_scMask, _mm256_set1_epi32(mismatch)));
		_S1 = _mm256_add_epi32(_S1, _tmp);

		_bcktrMask = _mm256_cmpgt_epi32(_F, _S1);
		_direction = _mm256_and_si256(_bcktrMask, _gap_h);
		_S1 = _mm256_max_epi32(_S1, _F);

		_bcktrMask = _mm256_cmpgt_epi32(_E, _S1);
		_direction = _mm256_blendv_epi8(_direction, _gap_v, _bcktrMask);
		_mm256_store_si256((__m256i*)(bcktrack + (query_length + padding - 1) * default_bw * band_count + (i - 1) * default_bw), _direction);
		_S1 = _mm256_max_epi32(_S1, _E);

		_tmp = _mm256_sub_epi32(_S1, _gapo);
		_E = _mm256_sub_epi32(_E, _gape);
		_bcktrMask = _mm256_cmpgt_epi32(_tmp, _E);
		_gap_v = _mm256_blendv_epi8(_mm256_add_epi32(_gap_v, _mm256_set1_epi32(1)), _mm256_set1_epi32(1), _bcktrMask);
		_mm256_store_si256((__m256i*)(gap + padding + query_length - 1 - (i - 1)), _gap_v);
		_E = _mm256_max_epi32(_tmp, _E);

		_F = _mm256_sub_epi32(_F, _gape);
		_bcktrMask = _mm256_cmpgt_epi32(_tmp, _F);
		_gap_h = _mm256_blendv_epi8(_mm256_sub_epi32(_gap_h, _mm256_set1_epi32(1)), _mm256_set1_epi32(-1), _bcktrMask);
		_F = _mm256_max_epi32(_tmp, _F);

		//save border values for the next band
		score[i + 1 - actual_bw] = _mm256_get_epi32(_S1, actual_bw - 1); // replace by inline function to extract from non-constant position, we'll need this score row to update ez later
		step[i + 1 - actual_bw] = _mm256_get_epi32(_E, actual_bw - 1);

		// save last column value for score
		_last_col = _mm256_add_epi32(_last_col, _mm256_and_si256(_hMask, _S1));

		// shift _S, _E, _hMask
		_S1 = _mm256_shift_left_si256<4>(_S1);
		_E = _mm256_shift_left_si256<4>(_E);
		_hMask = _mm256_shift_left_si256<4>(_hMask);

		_tmp = _S0; _S0 = _S1; _S1 = _tmp;
	}

#ifdef DEBUG_PRINT
	fprintf(stderr, "Score\n");
	print_array(stderr, score, ncol);
	fprintf(stderr, "Step_down\n");
	print_array(stderr, step, ncol);
	fprintf(stderr, "Gap_down\n");
	print_array(stderr, gap, query_length + 2 * padding);

	fprintf(stderr, "Backtrack matrix\n");
	print_bcktrMatrix(bcktrack, target_length, query_length, default_bw);
#endif

	// update last column max
	int last_col[8];
	_mm256_store_si256((__m256i*)&last_col, _last_col);

	for (i = 0; i < actual_bw; i++)
		if (last_col[i] >= ez->mqe)
		{
			ez->mqe_t = default_bw * band_count + i + 1;
			ez->mqe = last_col[i];
		}
}

int calculateCigar_avx(int *bcktrack, int n, int m, int bw, int overhangStrategy, ScoreMax *ez, string * cigar)
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
		btr = bcktrack[bcktrMatrix_index(I-1, J-1, altLength + bw - 1, bw)];
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


