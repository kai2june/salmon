#pragma once

#include "SalmonMath.hpp"
#include <atomic>
#include <mutex>
#include <vector>
#include <numeric>

class FragmentCoverageDistribution
{
  public:
    FragmentCoverageDistribution(size_t numBins = 100) 
        : numBins_(numBins), coeff_(numBins+1), count_(numBins+1), totCount_(0), pmf_(numBins+1), cmf_(numBins+1), isFinal(false) 
    {
        std::iota(coeff_.rbegin(), coeff_.rend(), 1);
    }

    void addCountFromSAM(size_t pos, size_t pnext, int32_t tlen, size_t reflen)
    {
        size_t start_bin(0), end_bin(0);
        computeBin(pos, pnext, tlen, reflen, start_bin, end_bin);
        count_[start_bin] += 1;
        count_[end_bin] -= 1;
        totCount_ += (end_bin - start_bin);
    }
    
    /// @brief evaluating how many base count a fragment spans and then divided by total base count  
    double evaluateProb(size_t pos, size_t pnext, int32_t tlen, size_t reflen)
    {
        size_t start_bin(0), end_bin(0);
        computeBin(pos, pnext, tlen, reflen, start_bin, end_bin);
        
        double prob = 0.0;
        for(size_t i=0; i<end_bin+1; ++i)
            prob += coeff_[numBins_ - end_bin + i]*count_[i];
        for(size_t i=0; i<start_bin; ++i)
            prob -= coeff_[numBins_ - (start_bin-1) + i]*count_[i];
        prob /= totCount_.load();

        return prob; 
    }

    void finalize() 
    {
        isFinal.store(true);
        finalizePMF();
        finalizeCMF();
    }
  public:
    const std::vector<std::atomic<int32_t>>& getCount()
    {
        return count_;
    }
    size_t getTotCount()
    {
        return totCount_.load();
    }
    const std::vector<std::atomic<double>>& getpmf()
    {
        return pmf_;
    }
    const std::vector<std::atomic<double>>& getcmf()
    {
        return cmf_;
    }
    bool getIsFinal()
    {
        return isFinal.load();
    }

  private:
    void computeBin(size_t pos, size_t pnext, int32_t tlen, size_t reflen, size_t& start_bin, size_t& end_bin)
    {
        size_t start = std::min(pos, pnext);
        size_t end = start + std::abs(tlen);
        double start_bin = (double)(start-1) / (double)reflen * (double)numBins_;
        if(start_bin > numBins_)
            start_bin = numBins_;
        double end_bin = (double)(end-1) / (double)reflen * (double)numBins_;
        if(end_bin > numBins_)
            end_bin = numBins_;
    }

    void finalizePMF()
    {
        for(size_t i=0; i<pmf_.size(); ++i)
            pmf_[i].store( std::accumulate(count_.begin(), count_.begin()+i+1, 0) / totCount_.load() );
    }

    void finalizeCMF()
    {
        for(size_t i=0; i<cmf_.size(); ++i)
            cmf_[i].store( std::accumulate(pmf_.begin(), pmf_.begin()+i+1, 0 ) );
    }

    size_t numBins_;

    /// @brief coefficient for calculating plusplusminusminus algorithm
    std::vector<size_t> coeff_;

    std::vector<std::atomic<int32_t>> count_;
    std::atomic<size_t> totCount_;

    std::vector<std::atomic<double>> pmf_;
    std::vector<std::atomic<double>> cmf_;

    std::atomic<bool> isFinal;
};