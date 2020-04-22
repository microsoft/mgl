#include "pairhmm_common.h"

using namespace std;

#define NUMBER float

static const float threeOver = 1.0f / 3.0f;

float compute_fast_prob_float(readinfo &read, vector<hapinfo> &hap_array)
{
    size_t numHaplotypes = hap_array.size();

    Context<float> ctx;

    int _c = read.c[0] & 127;
    float pGapM = 1.0f - ctx.ph2pr[_c];
    float distM = 1.0f - ctx.ph2pr[read.q[0] & 127];

    float resultMatch = distM * pGapM * ctx.INITIAL_CONSTANT;

    for (size_t r = 1; r < read.rslen; r++)
    {
        distM = 1.0f - ctx.ph2pr[read.q[r] & 127];

        int _i = read.i[r] & 127;
        int _d = read.d[r] & 127;
		float pMM = ctx.set_mm_prob(_i, _d);

        resultMatch *= distM * pMM;
    }

    char* rs = read.rs;
    size_t rsLen = read.rslen;

    for (size_t hap_idx = 0; hap_idx < numHaplotypes; ++hap_idx)
    {
        float resultHap = 0.0f;

        char* hap = hap_array[hap_idx].hap;
        size_t hapLen = hap_array[hap_idx].haplen;

        if (hapLen >= rsLen)
        {
            for (size_t hapPos = 0; hapPos <= hapLen - rsLen; hapPos++)
            {
                bool exactMatch = true;

                char* rs0 = rs;
                char* hap0 = hap + hapPos;

                for (size_t rsPos0 = 0; rsPos0 < rsLen; rsPos0++, rs0++, hap0++)
                {
                    if ((*rs0 != *hap0) && (*rs0 != 'N') && (*hap0 != 'N'))
                    {
                        exactMatch = false;

                        // Looking for single Gap
                        bool partialMatch = true;

                        char* rs1 = rs0 + 1;
                        char* hap1 = hap0 + 1;

                        for (size_t rsPos1 = rsPos0 + 1; rsPos1 < rsLen; rsPos1++, rs1++, hap1++)
                        {
                            if ((*rs1 != *hap1) && (*rs1 != 'N') && (*hap1 != 'N'))
                            {
                                partialMatch = false;

                                break;
                            }
                        }

                        if (partialMatch == true)
                        {
                            float dist = ctx.ph2pr[read.q[rsPos0] & 127];
                            float distM = 1.0f - dist;
                            float distN = dist * threeOver;

                            resultHap += resultMatch * distN / distM;
                        }

                        break;
                    }
                }

                if (exactMatch == true)
                {
                    resultHap += resultMatch;
                }
            }
        }

        hap_array[hap_idx].score = resultHap / hap_array[hap_idx].haplen;
    }

    return resultMatch;
}
