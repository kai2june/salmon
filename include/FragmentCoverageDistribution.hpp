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
        : numBins_(numBins), coeff_(numBins+1), count_(numBins+1), totCount_(0.0), pmf_(numBins+1), cmf_(numBins+1), isFinal_(false) 
    {
        std::iota(coeff_.rbegin(), coeff_.rend(), 1);
    }

    void addCountFromSAM(double mass, int32_t reflen, int32_t tlen, int32_t pos, int32_t pnext=std::numeric_limits<int32_t>::max())
    {
        int32_t start_bin(0), end_bin(0);
        computeBin(reflen, tlen, pos, pnext, start_bin, end_bin);
        double new_start = count_[start_bin].load() + mass;
        double new_end = count_[end_bin].load() - mass;
        double new_totCount = totCount_.load() + (end_bin-start_bin)*mass;
        count_[start_bin].store(new_start);
        count_[end_bin].store(new_end);
        totCount_.store(new_totCount);
    }
    
    /// @brief evaluating how many base count a fragment spans and then divided by total base count  
    double evaluateProb(int32_t reflen, int32_t tlen, int32_t pos, int32_t pnext=std::numeric_limits<int32_t>::max())
    {
        double prob = 0.0;
        int32_t start_bin(0), end_bin(0);
        computeBin(reflen, tlen, pos, pnext, start_bin, end_bin);
        if (isFinal_)
        {
            if(start_bin == 0)
                prob = cmf_[end_bin-1].load();
            else
                prob = cmf_[end_bin-1].load() - cmf_[start_bin-1].load();
        }
        else
        {
            for(int32_t i=0; i<end_bin; ++i)
                prob += (double)coeff_[numBins_ - (end_bin-1) + i] * count_[i].load();
            for(int32_t i=0; i<start_bin; ++i)
                prob -= (double)coeff_[numBins_ - (start_bin-1) + i] * count_[i].load();
            prob /= totCount_.load();
        }

        return prob; 
    }

    void finalize() 
    {
        isFinal_.store(true);
        finalizePMF();
        finalizeCMF();
    }
  public:
    const std::vector<std::atomic<double>>& getCount() const
    {
        return count_;
    }
    double getTotCount() const
    {
        return totCount_.load();
    }
    const std::vector<std::atomic<double>>& getpmf() const
    {
        return pmf_;
    }
    const std::vector<std::atomic<double>>& getcmf() const
    {
        return cmf_;
    }
    bool getIsFinal() const
    {
        return isFinal_.load();
    }

  private:
    /// @brief return closed-opened interval [start_bin, end_bin)
    void computeBin(int32_t reflen, int32_t tlen, int32_t pos, int32_t pnext, int32_t& start_bin, int32_t& end_bin)
    {
        int32_t start = std::min(pos, pnext);
        int32_t end = start + std::abs(tlen);
        start_bin = (int32_t)( (double)(start-1) / (double)reflen * (double)numBins_ );
        if(start_bin > numBins_)
            start_bin = numBins_;
        end_bin = (int32_t)( (double)(end-1) / (double)reflen * (double)numBins_ );
        if(end_bin > numBins_)
            end_bin = numBins_;
    }

    void finalizePMF()
    {
        for(size_t i=0; i<pmf_.size(); ++i)
            pmf_[i].store( std::accumulate(count_.begin(), count_.begin()+i+1, 0.0) / totCount_.load() );
    }

    void finalizeCMF()
    {
        for(size_t i=0; i<cmf_.size(); ++i)
            cmf_[i].store( std::accumulate(pmf_.begin(), pmf_.begin()+i+1, 0.0 ) );
    }

    size_t numBins_;

    /// @brief coefficient for calculating plusplusminusminus algorithm
    std::vector<int32_t> coeff_;

    std::vector<std::atomic<double>> count_;
    std::atomic<double> totCount_;

    std::vector<std::atomic<double>> pmf_;
    std::vector<std::atomic<double>> cmf_;

    std::atomic<bool> isFinal_;
};