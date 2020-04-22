#pragma once
#ifndef COMPUTE_PROB_SCALAR_H
#define COMPUTE_PROB_SCALAR_H

#include "pairhmm_common.h"

void compute_prob_scalarf(readinfo & read, std::vector<hapinfo>& hap_array);

void compute_prob_scalard(readinfo & read, std::vector<hapinfo>& hap_array);

#endif