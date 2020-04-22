#pragma once
#ifndef COMPUTE_PROB_AVX_H
#define COMPUTE_PROB_AVX_H

#include "pairhmm_common.h"

void compute_prob_avxf(readinfo & read, std::vector<hapinfo>& hap_array);

void compute_prob_avxd(readinfo & read, std::vector<hapinfo>& hap_array);

#endif