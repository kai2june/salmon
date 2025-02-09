/**
 *  lengthdistribution.cpp
 *  express
 *
 *  Created by Adam Roberts on 1/30/13.
 *  Copyright 2013 Adam Roberts. All rights reserved.
 *  Modified 2014, 2015, 2016, 2017 by Rob Patro.
 */

#include "FragmentLengthDistribution.hpp"
#include "SalmonMath.hpp"
#include "SalmonUtils.hpp"
#include <boost/assign.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace std;

FragmentLengthDistribution::FragmentLengthDistribution(
    double alpha, size_t max_val, double prior_mu, double prior_sigma,
    size_t kernel_n, double kernel_p, size_t bin_size)
    : /*hist_(max_val / bin_size + 1),*/ cachedCMF_(hist_.size()),
      haveCachedCMF_(false), totMass_(salmon::math::LOG_0),
      sum_(salmon::math::LOG_0), min_(max_val / bin_size), binSize_(bin_size) {

  using salmon::math::logAdd;
  max_val = max_val / bin_size;
  kernel_n = kernel_n / bin_size;
  assert(kernel_n % 2 == 0);

  double tot = log(alpha);

  // Set to prior distribution
  if (prior_mu > 0.0) {
    /// @brief bin_size=2 表示0bp,1bp屬同一length, 2bp,3bp屬同一length區間, ..., 248bp,249bp同一length區間
    boost::math::normal norm(prior_mu / bin_size,
                             prior_sigma / (bin_size * bin_size));

    std::vector<std::atomic<double>> hist_tmp(max_val / bin_size + 1); /// @brief +1是表示>1000bp則歸類在最後一個bin
    std::swap(hist_, hist_tmp);

    for (size_t i = 0; i <= max_val; ++i) {
      double norm_mass =
          boost::math::cdf(norm, i + 0.5) - boost::math::cdf(norm, i - 0.5); /// 表示(i-0.5, i+0.5)的pdf
      double mass = salmon::math::LOG_EPSILON; /// @brief ln(LOG_EPSILON) = -24.0067
      if (norm_mass != 0) {
        mass = tot + log(norm_mass); /// @brief alpha預設是1 i.e., log1=0, 不會加到
      }
      hist_[i].store(mass); /// @brief hist_就是observed    
      /// @brief norm_mass[458]似乎已達1.0
// std::cerr << "norm_mass=" << norm_mass << " hist_[" << i << "]=" << hist_[i].load() << std::endl;
      sum_.store(logAdd(sum_, log((double)i) + mass));
      totMass_.store(logAdd(totMass_, mass));
    }
  } else {
    std::vector<std::atomic<double>> hist_tmp(max_val + 1);
    std::swap(hist_, hist_tmp);
    std::fill(hist_.begin(), hist_.end(), tot - log((double)max_val));
    
    hist_[0].store(salmon::math::LOG_0);
    sum_.store(hist_[1] + log((double)(max_val * (max_val + 1))) - log(2.));
    totMass_ = tot;
  }

  // Define kernel
  boost::math::binomial_distribution<double> binom(kernel_n, kernel_p);
  kernel_ = vector<double>(kernel_n + 1);
  for (size_t i = 0; i <= kernel_n; i++) {
    kernel_[i] = log(boost::math::pdf(binom, i));
    // std::cerr << "kernel_[" << i << "]=" << kernel_[i] << std::endl;
    /// @brief logarithm: -2.77259, -1.38629, -0.980829, -1.38629, -2.77259
    /// @brief {0.0625,0.25,0.375,0.25,0.0625} = {1/16, 4/16, 6/16, 4/16, 1/16}
    /// @brief 1:4:6:4:1 即帕斯卡三角形
  }
}

size_t FragmentLengthDistribution::maxVal() const {
  return (hist_.size() - 1) * binSize_;
}

size_t FragmentLengthDistribution::minVal() const {
  if (min_ == hist_.size() - 1) {
    return 1;
  }
  return min_;
}

void FragmentLengthDistribution::addVal(size_t len, double mass) {
  using salmon::math::logAdd;
  // assert(!isnan(mass));
  // assert(kernel_.size());

  len /= binSize_;

  if (len > maxVal()) {
    len = maxVal();
  }
  if (len < min_) {
    min_ = len;
  }

  /// @brief https://dsp.stackexchange.com/questions/9987/gaussian-kernel-and-bionomial-coefficients
  /// @brief 這是pascal triangle的offset; offset=len-0時表該列中心點
  size_t offset = len - kernel_.size() / 2; 

  for (size_t i = 0; i < kernel_.size(); i++) {
    if (offset > 0 && offset < hist_.size()) {
      /// @brief mass是傳logForgettingMass; kernel_[i]也是取過log的此式等同兩者相乘
      double kMass = mass + kernel_[i]; 
      /// @brief 以len為中心, kernel_.size()為大小的window之中都要加上( mass+kernel_[i] )
      /// @brief 看到一條fragment長250bp, 則248~252bp都會加一些值
      salmon::utils::incLoopLog(hist_[offset], kMass); 
      /// @brief sum_表weighted length, log(offset)表對該長度取log, log(offset)+kMass表{linearspace之中weight*length}
      salmon::utils::incLoopLog(sum_, std::log(static_cast<double>(offset)) + kMass);
      salmon::utils::incLoopLog(totMass_, kMass);
    }
    offset++;
  }
}

/**
 * Returns the *LOG* probability of observing a fragment of length *len*.
 */
double FragmentLengthDistribution::pmf(size_t len) const {
  if (haveCachedCMF_) {
    return (len < cachedPMF_.size()) ? cachedPMF_[len] : cachedPMF_.back();
  } else {
    len /= binSize_;
    if (len > maxVal()) {
      len = maxVal();
    }
    return hist_[len] - totMass_;
  }
}

/**
 * Dumps the PMF to the provided vector.
 */
void FragmentLengthDistribution::dumpPMF(std::vector<double>& pmfOut,
                                         size_t& minV, size_t& maxV) const {

  minV = minVal();
  maxV = maxVal();
  pmfOut.clear();
  pmfOut.reserve(maxV - minV + 1);
  for (size_t i = minV; i <= maxV; ++i) {
    pmfOut.push_back(pmf(i));
  }
}

double FragmentLengthDistribution::cmf(size_t len) const {
  if (haveCachedCMF_) {
    return (len < cachedCMF_.size()) ? cachedCMF_[len] : cachedCMF_.back();
  } else {
    double cum = salmon::math::LOG_0;
    len /= binSize_;
    if (len > maxVal()) {
      len = maxVal();
    }

    for (size_t i = 0; i <= len; ++i) {
      cum = salmon::math::logAdd(cum, hist_[i]);
    }
    return cum - totMass_;
  }
}

std::vector<double> getLockedPMF(FragmentLengthDistribution* fld) {
  std::vector<double> pmfOut;
  auto maxV = fld->maxVal();
  pmfOut.reserve(maxV + 1);
  double totMass = salmon::math::LOG_0;
  for (size_t i = 0; i <= maxV; ++i) {
    pmfOut.push_back(fld->pmf(i));
    totMass = salmon::math::logAdd(totMass, pmfOut.back());
  }
  for (size_t i = 0; i <= maxV; ++i) {
    pmfOut[i] -= totMass;
  }
  return pmfOut;
}

void FragmentLengthDistribution::cacheCMF() {
  // std::lock_guard<std::mutex> lg(fldMut_);
  if (sl_.try_lock()) {
    if (!haveCachedCMF_) {
      size_t minV, maxV;
      cachedPMF_ = getLockedPMF(this);
      cachedCMF_ = cmf(cachedPMF_);
      haveCachedCMF_ = true;
    }

    sl_.unlock();
  }
}

/**
 * NOTE: It is *assumed* that pmf is properly normalized!
 **/
vector<double>
FragmentLengthDistribution::cmf(const std::vector<double>& pmf) const {
  double cum = salmon::math::LOG_0;
  double totalMass = salmon::math::LOG_0;
  vector<double> cdf(pmf.size());
  for (size_t i = 0; i < pmf.size(); ++i) {
    cum = salmon::math::logAdd(cum, pmf[i]);
    cdf[i] = cum;
  }
  // assert(approxEq(cum, totMass_));

  return cdf;
}

vector<double> FragmentLengthDistribution::cmf() const {
  double cum = salmon::math::LOG_0;
  vector<double> cdf(hist_.size());
  for (size_t i = 0; i < hist_.size(); ++i) {
    cum = salmon::math::logAdd(cum, hist_[i]);
    cdf[i] = cum - totMass_;
  }
  // assert(approxEq(cum, totMass_));

  return cdf;
}

double FragmentLengthDistribution::totMass() const { return totMass_; }

double FragmentLengthDistribution::mean() const { return sum_ - totMass(); }

std::string FragmentLengthDistribution::toString() const {
  std::stringstream ss;
  for (size_t i = 0; i < hist_.size(); ++i) {
    ss << std::exp(pmf(i * binSize_));
    if (i != hist_.size() - 1) {
      ss << '\t';
    }
  }
  ss << "\n";
  return ss.str();
}
/*
string FragmentLengthDistribution::to_string() const {
  string s = "";
  char buffer[50];
  for(size_t i = 0; i < hist_.size(); i++) {
    sprintf(buffer, "%e\t",sexp(pmf(i*binSize_)));
    s += buffer;
  }
  s.erase(s.length()-1,1);
  return s;
}

void FragmentLengthDistribution::append_output(ofstream& outfile,
                                       string length_type) const {
  outfile << ">" << length_type << " Length Distribution (0-" <<
maxVal()*binSize_; outfile << ")\n" << to_string() << endl;
}
*/
