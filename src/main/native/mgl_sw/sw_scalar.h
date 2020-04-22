#ifndef SW_SCALAR_H
#define SW_SCALAR_H

#include"sw_common.h"


void calculateMatrix(const char *target, int target_length, const char *query, int query_length, int *bcktrack, swParameters parameters, int overhangStrategy, ScoreMax *ez);
int calculateCigar(int *bcktrack, int n, int m, int overhangStrategy, ScoreMax * ez, std::string * cigar);
int align_scalar(const char *tseq, int target_length, const char *qseq, int query_length, swParameters parameters, int strategy, std::string * result_cigar);

#endif
