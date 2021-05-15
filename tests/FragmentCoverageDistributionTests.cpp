#include "FragmentCoverageDistribution.hpp"

SCENARIO("test FragmentCoverageDistribution.hpp")
{
    size_t numBins=100;
    FragmentCoverageDistribution fcd(numBins);
    fcd.addCountFromSAM(1, 201, 300, 1000); /// 1~300, 佔[0,30)th bin, 共30個bin
    REQUIRE( fcd.evaluateProb(1, 201, 300, 1000, false) == (1.0) );
    fcd.addCountFromSAM(301, 401, 200, 1000); /// 301~500, 佔[30,50)th bin, 共20個bin
    REQUIRE( fcd.evaluateProb(1, 201, 300, 1000, false) == (30.0/50.0) ); /// 101~300, 佔[0, 30) bin, 共30個bin

    std::vector<int32_t> ans_count(numBins+1, 0); 
    ans_count[0] += 1;
    ans_count[30] += -1;
    ans_count[30] += 1;
    ans_count[50] += -1;
    std::vector<double> ans_pmf(numBins+1, 0.0);
    for(int32_t i=0; i<50; ++i)
        ans_pmf[i] += 1.0 / 50.0;
    std::vector<double> ans_cmf(numBins+1, 0.0);
    ans_cmf[0] = ans_pmf[0];
    for(size_t i=1; i<ans_cmf.size(); ++i)
        ans_cmf[i] = ans_cmf[i-1] + ans_pmf[i];

    fcd.finalize();
    REQUIRE(fcd.getTotCount() == 50);
    const auto& count = fcd.getCount();
    for (size_t i=0; i<count.size(); ++i)
        REQUIRE(ans_count[i] == count[i].load());
    const auto& pmf = fcd.getpmf();
    for (size_t i=0; i<pmf.size(); ++i)
        REQUIRE(ans_pmf[i] == pmf[i].load());
    const auto& cmf = fcd.getcmf();
    for (size_t i=0; i<cmf.size(); ++i)
        REQUIRE(ans_cmf[i] == cmf[i].load());
}