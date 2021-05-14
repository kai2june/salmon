#include "FragmentCoverageDistribution.hpp"

SCENARIO("test FragmentCoverageDistribution.hpp")
{
    size_t numBins=100;
    FragmentCoverageDistribution fcd(numBins);
    fcd.addCountFromSAM(100, 200, 200, 1000); /// 佔2個bin
    fcd.addCountFromSAM(201, 301, 301, 1000); /// 佔4個bin
    REQUIRE( fcd.evaluateProb(100, 200, 200, 1000) == (1.0/3.0) ); /// 是0.33

    std::vector<int32_t> ans_count(numBins, 0); /// {0,1,2,1,1,1,0,0,0,0,...,0}, size=101
    ans_count[1] = 1;
    ans_count[2] = 1;
    ans_count[3] = -1;
    ans_count[6] = -1;
    std::vector<double> ans_pmf(101, 0.0);
    ans_pmf[1] = 1.0/6.0;
    ans_pmf[2] = 2.0/6.0;
    ans_pmf[3] = 1.0/6.0;
    ans_pmf[4] = 1.0/6.0;
    ans_pmf[5] = 1.0/6.0;
    std::vector<double> ans_cmf(101, 1.0);
    ans_cmf[0] = 0.0;
    ans_cmf[1] = 1.0/6.0;
    ans_cmf[2] = 3.0/6.0;
    ans_cmf[3] = 4.0/6.0;
    ans_cmf[4] = 5.0/6.0;

    fcd.finalize();
    REQUIRE(fcd.getTotCount(), 6);
    const auto& count = fcd.getCount();
    for (size_t i=0; i<count.size(); ++i)
        REQUIRE(ans_count[i], count[i].load());
    const auto& pmf = fcd.getpmf();
    for (size_t i=0; i<pmf.size(); ++i)
        REQUIRE(ans_pmf[i], pmf[i].load());
    const auto& cmf = fcd.getcmf();
    for (size_t i=0; i<cmf.size(); ++i)
        REQUIRE(ans_cmf[i], cmf[i].load());
}