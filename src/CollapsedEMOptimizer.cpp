#include <atomic>
#include <unordered_map>
#include <vector>
#include <exception>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_for_each.h"
#include "tbb/parallel_reduce.h"
#include "tbb/partitioner.h"
// <-- deprecated in TBB --> #include "tbb/task_scheduler_init.h"
#include "tbb/global_control.h"

//#include "fastapprox.h"
#include <boost/math/special_functions/digamma.hpp>

// C++ string formatting library
#include "spdlog/fmt/fmt.h"

#include "Eigen/Dense"
#include "cuckoohash_map.hh"

#include "AlignmentLibrary.hpp"
#include "BootstrapWriter.hpp"
#include "CollapsedEMOptimizer.hpp"
#include "MultinomialSampler.hpp"
#include "ReadExperiment.hpp"
#include "ReadPair.hpp"
#include "SalmonMath.hpp"
#include "Transcript.hpp"
#include "TranscriptGroup.hpp"
#include "UnpairedRead.hpp"
#include "EMUtils.hpp"

using BlockedIndexRange = tbb::blocked_range<size_t>;

// intelligently chosen value originally adopted from
// https://github.com/pachterlab/kallisto/blob/master/src/EMAlgorithm.h#L18
// later modified since denorm_min seems to be too permissive.
constexpr double minEQClassWeight = std::numeric_limits<double>::min();
constexpr double minWeight = std::numeric_limits<double>::min();
// A bit more conservative of a minimum as an argument to the digamma function.
constexpr double digammaMin = 1e-10;

double normalize(std::vector<std::atomic<double>>& vec) {


  double sum{0.0};
  for (auto& v : vec) {
    sum += v;
  }

  // too small!
  if (sum < ::minWeight) {
    return sum;
  }

  double invSum = 1.0 / sum;
  for (auto& v : vec) {
    v.store(v.load() * invSum);
  }

  return sum;
}

template <typename VecT>
double truncateCountVector(VecT& alphas, std::vector<double>& cutoff) {


  // Truncate tiny expression values
  double alphaSum = 0.0;

  for (size_t i = 0; i < alphas.size(); ++i) {
    if (alphas[i] <= cutoff[i]) {
      alphas[i] = 0.0;
    }
    alphaSum += alphas[i];
  }
  return alphaSum;
}

/**
 *  Populate the prior parameters for the VBEM
 *  Note: effLens *must* be valid before calling this function.
 */
/// @brief 從純量0.01展開成向量
/// @brief 設定perXXXPrior為priorValue
/// @brief perNucleotide的prior比perTranscript大[長度]倍
/// @brief priorValue : default 0.01
/// @brief priorAlphas會跟alpha一起用到, 加在一起
/// @brief perTranscriptPrior=0.01, perNucleotidePrior=0.01*~li
std::vector<double> populatePriorAlphas_(
    std::vector<Transcript>& transcripts, // transcripts
    Eigen::VectorXd& effLens,             // current effective length estimate
    double priorValue,      // the per-nucleotide prior value to use
    bool perTranscriptPrior // true if prior is per-txp, else per-nucleotide
) {


  // start out with the per-txp prior
  std::vector<double> priorAlphas(transcripts.size(), priorValue);

  // If the prior is per-nucleotide (default, then we need a potentially
  // different value for each transcript based on its length).
  if (!perTranscriptPrior) {
    for (size_t i = 0; i < transcripts.size(); ++i) {
      priorAlphas[i] = priorValue * effLens(i); /// @brief perNucleotide的prior比perTranscript大[長度]倍
    }
  }

//   for (size_t i(0); i<transcripts.size(); ++i)
//   {
//     double uniqueCount = static_cast<double>(transcripts[i].uniqueCount() + 0.5);
//     priorAlphas[i] *= uniqueCount;
//   }

  return priorAlphas;
}

/**
 * Single-threaded VBEM-update routine for use in bootstrapping
 */
template <typename VecT>
void VBEMUpdate_(std::vector<std::vector<uint32_t>>& txpGroupLabels,
                 std::vector<std::vector<double>>& txpGroupCombinedWeights,
                 const std::vector<uint64_t>& txpGroupCounts,
                 std::vector<double>& priorAlphas, 
                 const VecT& alphaIn, VecT& alphaOut, VecT& expTheta) {

  assert(alphaIn.size() == alphaOut.size());
  size_t M = alphaIn.size();
  size_t numEQClasses = txpGroupLabels.size();
  double alphaSum = {0.0};
  for (size_t i = 0; i < M; ++i) {
    alphaSum += alphaIn[i] + priorAlphas[i];
  }

  double logNorm = boost::math::digamma(alphaSum);

  // double prior = priorAlpha;

  for (size_t i = 0; i < M; ++i) {
    auto ap = alphaIn[i] + priorAlphas[i];
    if (ap > ::digammaMin) {
        expTheta[i] =
          std::exp(boost::math::digamma(ap) - logNorm);
    } else {
      expTheta[i] = 0.0;
    }
    alphaOut[i] = 0.0; // priorAlphas[i];
  }

  for (size_t eqID = 0; eqID < numEQClasses; ++eqID) {
    uint64_t count = txpGroupCounts[eqID];
    const std::vector<uint32_t>& txps = txpGroupLabels[eqID];
    const auto& auxs = txpGroupCombinedWeights[eqID];

    size_t groupSize = txpGroupCombinedWeights[eqID].size(); // txps.size();
    // If this is a single-transcript group,
    // then it gets the full count.  Otherwise,
    // update according to our VBEM rule.
    if (BOOST_LIKELY(groupSize > 1)) {
      double denom = 0.0;
      for (size_t i = 0; i < groupSize; ++i) {
        auto tid = txps[i];
        auto aux = auxs[i];
        if (expTheta[tid] > 0.0) {
          double v = expTheta[tid] * aux;
          denom += v;
        }
      }
      if (denom <= ::minEQClassWeight) {
        // tgroup.setValid(false);
      } else {
        double invDenom = count / denom;
        for (size_t i = 0; i < groupSize; ++i) {
          auto tid = txps[i];
          auto aux = auxs[i];
          if (expTheta[tid] > 0.0) {
            double v = expTheta[tid] * aux;
            salmon::utils::incLoop(alphaOut[tid], v * invDenom);
          }
        }
      }

    } else {
      salmon::utils::incLoop(alphaOut[txps.front()], count);
    }
  }
}


template <typename EQVecT>
void winnerTakesAll(EQVecT& eqVec,
                    CollapsedEMOptimizer::VecType& alphaIn,
                    CollapsedEMOptimizer::VecType& alphaOut,
                    double threshold)
{
        for (auto& kv : eqVec) {

          uint64_t count = kv.second.count;  
          // for each transcript in this class
          const TranscriptGroup& tgroup = kv.first;
          if (tgroup.valid) {
            const std::vector<uint32_t>& txps = tgroup.txps;
            const auto& auxs = kv.second.combinedWeights;

            size_t groupSize = kv.second.weights.size(); // txps.size();
            // If this is a single-transcript group,
            // then it gets the full count.  Otherwise,
            // update according to our VBEM rule.

            if (BOOST_LIKELY(groupSize > 1)) {

              double denom = 0.0;
              for (size_t i = 0; i < groupSize; ++i) {
                auto tid = txps[i];
                auto aux = auxs[i];
                double v = (alphaIn[tid]) * aux;
                denom += v;
              }

              if (denom <= ::minEQClassWeight) {
                // tgroup.setValid(false);
              } else {
                  if (*std::max_element(auxs.begin(), auxs.end()) > threshold)
                  {
                    if (!std::isnan(count)) {
                        salmon::utils::incLoop(alphaOut[txps[std::max_element(auxs.begin(), auxs.end())-auxs.begin()]], count);
                    }
                  }
                  else
                  {
                    double invDenom = count / denom;
                    for (size_t i = 0; i < groupSize; ++i) {
                        auto tid = txps[i];
                        auto aux = auxs[i];
                        double v = (alphaIn[tid]) * aux;
                        if (!std::isnan(v)) {
                            salmon::utils::incLoop(alphaOut[tid], v * invDenom);
                        }
                    }
                  }
              }
            } 
            else
            {
                salmon::utils::incLoop(alphaOut[txps.front()], count);
            }
          } /// @brief if(tgroup.valid)
        } /// @brief for eqID in eqVec.size()
}

template <typename EQVecT>
void eqClassInnerEntropy(EQVecT& eqVec,
                         CollapsedEMOptimizer::VecType& alphaIn,
                         CollapsedEMOptimizer::VecType& alphaOut)
{    
        for (auto& kv : eqVec) {

          double count = (double)kv.second.count;  
          // for each transcript in this class
          const TranscriptGroup& tgroup = kv.first;
          if (tgroup.valid) {
            const std::vector<uint32_t>& txps = tgroup.txps;
            const auto& auxs = kv.second.combinedWeights;

            size_t groupSize = kv.second.weights.size(); // txps.size();
            // If this is a single-transcript group,
            // then it gets the full count.  Otherwise,
            // update according to our VBEM rule.

            double entropy(0.0);
            for (size_t i = 0; i < groupSize; ++i)
                entropy += -auxs[i]*salmon::math::log(auxs[i]);
            if(entropy < 1.0)
                entropy = 0.33;
            std::cerr << "count=" << count << ",entropy=" << entropy << std::endl;
            if ( *std::max_element(auxs.begin(), auxs.end()) <= 0.5)
                count = count / entropy;

            if (BOOST_LIKELY(groupSize > 1)) {

              double denom = 0.0;
              for (size_t i = 0; i < groupSize; ++i) {
                auto tid = txps[i];
                auto aux = auxs[i];
                double v = (alphaIn[tid]) * aux;
                denom += v;
              }

              if (denom <= ::minEQClassWeight) {
                // tgroup.setValid(false);
              } else {
                    double invDenom = count / denom;
                    for (size_t i = 0; i < groupSize; ++i) 
                    {
                        auto tid = txps[i];
                        auto aux = auxs[i];
                        double v = (alphaIn[tid]) * aux;
                        if (!std::isnan(v)) {
                            salmon::utils::incLoop(alphaOut[tid], v * invDenom);
                        }
                    }
                  }
            }
            else
            {
                salmon::utils::incLoop(alphaOut[txps.front()], count);
            }
          } 
        } /// @brief if(tgroup.valid)
} /// @brief for eqID in eqVec.size()

/*
 * Use the "standard" EM algorithm over equivalence
 * classes to estimate the latent variables (alphaOut)
 * given the current estimates (alphaIn).
 */
/// @brief eqVec : vec<pair<TranscriptGroup, TranscriptValue>>
/// @brief priorAlphas : 0.01
/// @brief 上面修改alphas是一定比例projectedCount+一定比例uniqueCount
/// @brief alphasPrime : 1.0
template <typename EQVecT>
void EMUpdate_(EQVecT& eqVec,
               std::vector<double>& priorAlphas,
               const CollapsedEMOptimizer::VecType& alphaIn,
               CollapsedEMOptimizer::VecType& alphaOut,
               std::vector<double>& multimappedFrac) {

  assert(alphaIn.size() == alphaOut.size());

  tbb::parallel_for(
      BlockedIndexRange(size_t(0), size_t(eqVec.size())),
      [&eqVec, &priorAlphas, &alphaIn, &alphaOut, &multimappedFrac](const BlockedIndexRange& range) -> void {
        for (auto eqID : boost::irange(range.begin(), range.end())) {
          auto& kv = eqVec[eqID];

          /// @brief count表實際上mapped fragment數, weight表目前分配給各transcript比例
          uint64_t count = kv.second.count;  /// @brief 根據下面的code, count應該是{d^j}
          // for each transcript in this class
          const TranscriptGroup& tgroup = kv.first;
if (tgroup.valid) {
            const std::vector<uint32_t>& txps = tgroup.txps;
            /// @brief combinedWeights : count * weight * probStartPos
            const auto& auxs = kv.second.combinedWeights;

            size_t groupSize = kv.second.weights.size(); // txps.size();
            // If this is a single-transcript group,
            // then it gets the full count.  Otherwise,
            // update according to our VBEM rule.

            if (BOOST_LIKELY(groupSize > 1)) {
              double denom = 0.0;
for (size_t i = 0; i < groupSize; ++i) {
                auto tid = txps[i];
// if (tid == 11821) std::cerr << "groupSize:" << groupSize << std::endl;
// if (tid == 11822) std::cerr << "groupSize:" << groupSize << std::endl;
                auto aux = auxs[i]*multimappedFrac[tid];
                /// @brief aux是論文的{w^j_i}, 即combinedWeight
                /// @brief v是equation11的分子(不含{d^j})
                /// @brief v就是Estep算的東西
                double v = (alphaIn[tid]) * aux;
                /// @brief denom是論文equation11的分母
                denom += v;
}

              if (denom <= ::minEQClassWeight) {
                // tgroup.setValid(false);
              } else {
                /// @brief count就是{d^j}
                double invDenom = count / denom;
for (size_t i = 0; i < groupSize; ++i) {
                  auto tid = txps[i];
                  auto aux = auxs[i]*multimappedFrac[tid];
                  double v = (alphaIn[tid]) * aux;
                  if (!std::isnan(v)) {
                    /// @brief Mstep更新alpha
// if (tid == 31744) std::cerr << "old_alphaIn[FBtr0300835]: " << alphaIn[tid] << " old_alphaOut[FBtr0300835]" << alphaOut[tid] << " v*invDenom[FBtr0300835]: " << v*invDenom << " combinedWeights:" << aux << " count: " << count << std::endl;
// if (tid == 31745) std::cerr << "old_alphaIn[FBtr0300834]: " << alphaIn[tid] << " old_alphaOut[FBtr0300834]" << alphaOut[tid] << " v*invDenom[FBtr0300834]: " << v*invDenom << " combinedWeights:" << aux << " count: " << count << std::endl;
// if (tid == 15301) std::cerr << "old_alphaIn[FBtr0299940]: " << alphaIn[tid] << " old_alphaOut[FBtr0299940]" << alphaOut[tid] << " v*invDenom[FBtr0299940]: " << v*invDenom << " combinedWeights:" << aux << " count: " << count << std::endl;
// if (tid == 15302) std::cerr << "old_alphaIn[FBtr0299941]: " << alphaIn[tid] << " old_alphaOut[FBtr0299941]" << alphaOut[tid] << " v*invDenom[FBtr0299941]: " << v*invDenom << " combinedWeights:" << aux << " count: " << count << std::endl;
                    salmon::utils::incLoop(alphaOut[tid], v * invDenom);
// if (tid == 31744) std::cerr << "alphaOut[FBtr0300835]: " << alphaOut[tid] << std::endl;
// if (tid == 31745) std::cerr << "alphaOut[FBtr0300834]: " << alphaOut[tid] << std::endl;
// if (tid == 15301) std::cerr << "alphaOut[FBtr0299940]: " << alphaOut[tid] << std::endl;
// if (tid == 15302) std::cerr << "alphaOut[FBtr0299941]: " << alphaOut[tid] << std::endl;
                  }
}
              }
            } else {
auto tid = txps[txps.front()];
// if (tid == 11821) std::cerr << "groupSize:" << groupSize << std::endl;
// if (tid == 11822) std::cerr << "groupSize:" << groupSize << std::endl;
// if (tid == 11821) std::cerr << "old_alphaIn[FBtr0086273]: " << alphaIn[tid] << " count: " << count << std::endl;
// if (tid == 11822) std::cerr << "old_alphaIn[FBtr0086274]: " << alphaIn[tid] << " count: " << count << std::endl;
              salmon::utils::incLoop(alphaOut[txps.front()], count);
// if (tid == 11821) std::cerr << "alphaOut[FBtr0086273]: " << alphaOut[tid] << std::endl;
// if (tid == 11822) std::cerr << "alphaOut[FBtr0086274]: " << alphaOut[tid] << std::endl;
            }
            // if(eqID==118 || eqID==15536 || eqID==20227)std::cerr << "eqID:" << eqID << "alphaOut[FBtr0086273]: " << alphaOut[11821] << "_groupSize:" << groupSize << "_count:" << count << std::endl;
            // if(eqID==118 || eqID==15536 || eqID==20227)std::cerr << "eqID:" << eqID << "alphaOut[FBtr0086274]: " << alphaOut[11822] << "_groupSize:" << groupSize << "_count:" << count << std::endl;
} /// @brief if(tgroup.valid)
        } /// @brief for eqID in eqVec.size()
      });
} /// @brief EMupdate_ end

/*
 * Use the Variational Bayesian EM algorithm over equivalence
 * classes to estimate the latent variables (alphaOut)
 * given the current estimates (alphaIn).
 */
template <typename EQVecT>
void VBEMUpdate_(EQVecT& eqVec,
                 std::vector<double>& priorAlphas, 
                 const CollapsedEMOptimizer::VecType& alphaIn,
                 CollapsedEMOptimizer::VecType& alphaOut,
                 CollapsedEMOptimizer::VecType& expTheta,
                 std::vector<double>& multimappedFrac) {

/// @brief 公式: digamma(alpha_i*priorAlpha_i) / Sigma{digamma(alpha_i*priorAlpha_i)} 
  assert(alphaIn.size() == alphaOut.size());
  size_t M = alphaIn.size();
  double alphaSum = {0.0};
  for (size_t i = 0; i < M; ++i) {
    alphaSum += alphaIn[i] + priorAlphas[i];
  }

  double logNorm = boost::math::digamma(alphaSum);

  tbb::parallel_for(BlockedIndexRange(size_t(0), size_t(priorAlphas.size())),
                    [logNorm, &priorAlphas, &alphaIn, &alphaOut,
                     &expTheta](const BlockedIndexRange& range) -> void {

                      // double prior = priorAlpha;

                      for (auto i : boost::irange(range.begin(), range.end())) {
                        auto ap = alphaIn[i].load() + priorAlphas[i];
                        /// @brief expTheta[i] = digamma(alphaIn[i] + priorAlphas[i]) - digamma(alphaIn[:] + priorAlphas[:])
                        if (ap > ::digammaMin) {
                          expTheta[i] =
                              std::exp(boost::math::digamma(ap) - logNorm);
                        } else {
                          expTheta[i] = 0.0;
                        }
                        // alphaOut[i] = prior * transcripts[i].RefLength;
                        alphaOut[i] = 0.0;
                      }
                    });

  tbb::parallel_for(
      BlockedIndexRange(size_t(0), size_t(eqVec.size())),
      [&eqVec, &alphaIn, &alphaOut,
       &expTheta, &multimappedFrac](const BlockedIndexRange& range) -> void {
        for (auto eqID : boost::irange(range.begin(), range.end())) {
          auto& kv = eqVec[eqID];

          uint64_t count = kv.second.count;
          // for each transcript in this class
          const TranscriptGroup& tgroup = kv.first;
          if (tgroup.valid) {
            const std::vector<uint32_t>& txps = tgroup.txps;
            const auto& auxs = kv.second.combinedWeights;

            size_t groupSize = kv.second.weights.size(); // txps.size();
            // If this is a single-transcript group,
            // then it gets the full count.  Otherwise,
            // update according to our VBEM rule.

            if (BOOST_LIKELY(groupSize > 1)) {
              double denom = 0.0;
              for (size_t i = 0; i < groupSize; ++i) {
                auto tid = txps[i];
// if (tid == 11821) std::cerr << "groupSize:" << groupSize << std::endl;
// if (tid == 11822) std::cerr << "groupSize:" << groupSize << std::endl;
                auto aux = auxs[i]*multimappedFrac[tid];
                if (expTheta[tid] > 0.0) {
                  double v = expTheta[tid] * aux;
                  denom += v;
                }
              }
              if (denom <= ::minEQClassWeight) {
                // tgroup.setValid(false);
              } else {
                double invDenom = count / denom;
                for (size_t i = 0; i < groupSize; ++i) {
                  auto tid = txps[i];
                  auto aux = auxs[i]*multimappedFrac[tid];
                  
                  if (expTheta[tid] > 0.0) {
                    double v = expTheta[tid] * aux;
// if (tid == 15299) std::cerr << "old_alphaIn[FBtr0091710]:" << alphaIn[tid] << " old_alphaOut[FBtr0091710]" << alphaOut[tid] << " v*invDenom[FBtr0091710]:" << v*invDenom << " combinedWeights:" << aux << " count:" << count << std::endl;
// if (tid == 15300) std::cerr << "old_alphaIn[FBtr0091711]:" << alphaIn[tid] << " old_alphaOut[FBtr0091711]" << alphaOut[tid] << " v*invDenom[FBtr0091711]:" << v*invDenom << " combinedWeights:" << aux << " count:" << count << std::endl;
// if (tid == 15301) std::cerr << "old_alphaIn[FBtr0299940]: " << alphaIn[tid] << " old_alphaOut[FBtr0299940]" << alphaOut[tid] << " v*invDenom[FBtr0299940]: " << v*invDenom << " combinedWeights:" << aux << " count: " << count << std::endl;
// if (tid == 15302) std::cerr << "old_alphaIn[FBtr0299941]: " << alphaIn[tid] << " old_alphaOut[FBtr0299941]" << alphaOut[tid] << " v*invDenom[FBtr0299941]: " << v*invDenom << " combinedWeights:" << aux << " count: " << count << std::endl;
                    salmon::utils::incLoop(alphaOut[tid], v * invDenom);
// if (tid == 15299) std::cerr << "alphaOut[FBtr0091710]:" << alphaOut[tid] << std::endl;
// if (tid == 15300) std::cerr << "alphaOut[FBtr0091711]:" << alphaOut[tid] << std::endl;
// if (tid == 15301) std::cerr << "alphaOut[FBtr0299940]: " << alphaOut[tid] << std::endl;
// if (tid == 15302) std::cerr << "alphaOut[FBtr0299941]: " << alphaOut[tid] << std::endl;
                  }
                }
              }

            } else {
// auto tid = txps[txps.front()];
// if (tid == 11821) std::cerr << "groupSize:" << groupSize << std::endl;
// if (tid == 11822) std::cerr << "groupSize:" << groupSize << std::endl;
// if (tid == 11821) std::cerr << "old_alphaIn[FBtr0086273]: " << alphaIn[tid] << " count: " << count << std::endl;
// if (tid == 11822) std::cerr << "old_alphaIn[FBtr0086274]: " << alphaIn[tid] << " count: " << count << std::endl;
              salmon::utils::incLoop(alphaOut[txps.front()], count);
// if (tid == 11821) std::cerr << "alphaOut[FBtr0086273]: " << alphaOut[tid] << std::endl;
// if (tid == 11822) std::cerr << "alphaOut[FBtr0086274]: " << alphaOut[tid] << std::endl;
            }
            // if(eqID==118 || eqID==15536 || eqID==20227) std::cerr << "eqID:" << eqID << "alphaOut[FBtr0086273]: " << alphaOut[11821] << std::endl;
            // if(eqID==118 || eqID==15536 || eqID==20227) std::cerr << "eqID:" << eqID << "alphaOut[FBtr0086274]: " << alphaOut[11822] << std::endl;
} /// @brief if tgroup.valid
        } /// @brief for eqID in eqVec.size()
      });
}

template <typename VecT, typename EQVecT>
size_t markDegenerateClasses(
    EQVecT& eqVec,
    VecT& alphaIn, std::vector<bool>& available,
    std::shared_ptr<spdlog::logger> jointLog, bool verbose = false) {

  size_t numDropped{0};
for (auto& kv : eqVec) {
    uint64_t count = kv.second.count;
    // for each transcript in this class
    const TranscriptGroup& tgroup = kv.first;
    const std::vector<uint32_t>& txps = tgroup.txps;
    const auto& auxs = kv.second.combinedWeights;

    double denom = 0.0;
    size_t groupSize = kv.second.weights.size();
for (size_t i = 0; i < groupSize; ++i) {
      auto tid = txps[i];
      auto aux = auxs[i];
      double v = alphaIn[tid] * aux;
      if (!std::isnan(v)) {
        denom += v;
      } else {
        std::cerr << "val is NAN; alpha( " << tid << " ) = " << alphaIn[tid]
                  << ", aux = " << aux << "\n";
      }
}
    if (denom <= ::minEQClassWeight) {
      fmt::MemoryWriter errstream;

      errstream << "\nDropping weighted eq class\n";
      errstream << "============================\n";

      errstream << "denom = 0, count = " << count << "\n";
      errstream << "class = { ";
      for (size_t tn = 0; tn < groupSize; ++tn) {
        errstream << txps[tn] << " ";
      }
      errstream << "}\n";
      errstream << "alphas = { ";
      for (size_t tn = 0; tn < groupSize; ++tn) {
        errstream << alphaIn[txps[tn]] << " ";
      }
      errstream << "}\n";
      errstream << "weights = { ";
      for (size_t tn = 0; tn < groupSize; ++tn) {
        errstream << auxs[tn] << " ";
      }
      errstream << "}\n";
      errstream << "============================\n\n";

      if (verbose) {
        jointLog->info(errstream.str());
      }
      ++numDropped;
      kv.first.setValid(false);
    } else {
      for (size_t i = 0; i < groupSize; ++i) {
        auto tid = txps[i];
        available[tid] = true;
      }
    }
} /// @brief for kv in eqVec
  return numDropped;
}

CollapsedEMOptimizer::CollapsedEMOptimizer() {}

bool doBootstrap(
    std::vector<std::vector<uint32_t>>& txpGroups,
    std::vector<std::vector<double>>& txpGroupCombinedWeights,
    std::vector<Transcript>& transcripts, Eigen::VectorXd& effLens,
    const std::vector<double>& sampleWeights, std::vector<uint64_t>& origCounts,
    uint64_t totalNumFrags,
    uint64_t numMappedFrags, double uniformTxpWeight,
    std::atomic<uint32_t>& bsNum, SalmonOpts& sopt,
    std::vector<double>& priorAlphas,
    std::function<bool(const std::vector<double>&)>& writeBootstrap,
    double relDiffTolerance, uint32_t maxIter) {


  // An EM termination criterion, adopted from Bray et al. 2016
  uint32_t minIter = 50;

  // Determine up front if we're going to use scaled counts.
  bool useScaledCounts = !(sopt.useQuasi or sopt.allowOrphans);
  bool useVBEM{sopt.useVBOpt};
  size_t numClasses = txpGroups.size();
  CollapsedEMOptimizer::SerialVecType alphas(transcripts.size(), 0.0);
  CollapsedEMOptimizer::SerialVecType alphasPrime(transcripts.size(), 0.0);
  CollapsedEMOptimizer::SerialVecType expTheta(transcripts.size(), 0.0);
  std::vector<uint64_t> sampCounts(numClasses, 0);

  uint32_t numBootstraps = sopt.numBootstraps;
  bool perTranscriptPrior{sopt.perTranscriptPrior};

  auto& jointLog = sopt.jointLog;

  #if defined(__linux) && defined(__GLIBCXX__) && __GLIBCXX__ >= 20200128
    std::random_device rd("/dev/urandom");
  #else
    std::random_device rd;
  #endif  // defined(__GLIBCXX__) && __GLIBCXX__ >= 2020012

  std::mt19937 gen(rd());
  // MultinomialSampler msamp(rd);
  std::discrete_distribution<uint64_t> csamp(sampleWeights.begin(),
                                             sampleWeights.end());
  while (bsNum++ < numBootstraps) {
    csamp.reset();

    for (size_t sc = 0; sc < sampCounts.size(); ++sc) {
      sampCounts[sc] = 0;
    }
    for (size_t fn = 0; fn < totalNumFrags; ++fn) {
      ++sampCounts[csamp(gen)];
    }
    // Do a new bootstrap
    // msamp(sampCounts.begin(), totalNumFrags, numClasses,
    // sampleWeights.begin());

    double totalLen{0.0};
    for (size_t i = 0; i < transcripts.size(); ++i) {
      alphas[i] =
          transcripts[i].getActive() ? uniformTxpWeight * totalNumFrags : 0.0;
      totalLen += effLens(i);
    }

    bool converged{false};
    double maxRelDiff = -std::numeric_limits<double>::max();
    size_t itNum = 0;

    // If we use VBEM, we'll need the prior parameters
    // double priorAlpha = 1.00;

    // EM termination criteria, adopted from Bray et al. 2016
    double minAlpha = 1e-8;
    double alphaCheckCutoff = 1e-2;
    double cutoff = minAlpha;

    while (itNum < minIter or (itNum < maxIter and !converged)) {

      if (useVBEM) {
        VBEMUpdate_(txpGroups, txpGroupCombinedWeights, sampCounts, 
                    priorAlphas, alphas, alphasPrime, expTheta);
      } else {
        EMUpdate_(txpGroups, txpGroupCombinedWeights, sampCounts, 
                  alphas, alphasPrime);
      }

      converged = true;
      maxRelDiff = -std::numeric_limits<double>::max();
      for (size_t i = 0; i < transcripts.size(); ++i) {
        if (alphasPrime[i] > alphaCheckCutoff) {
          double relDiff =
              std::abs(alphas[i] - alphasPrime[i]) / alphasPrime[i];
          maxRelDiff = (relDiff > maxRelDiff) ? relDiff : maxRelDiff;
          if (relDiff > relDiffTolerance) {
            converged = false;
          }
        }
        alphas[i] = alphasPrime[i];
        alphasPrime[i] = 0.0;
      }

      ++itNum;
    }

    // Consider the projection of the abundances onto the *original* equivalence class
    // counts
    if (sopt.bootstrapReproject) {
      if (useVBEM) {
        VBEMUpdate_(txpGroups, txpGroupCombinedWeights, origCounts, 
                    priorAlphas, alphas, alphasPrime, expTheta);
      } else {
        EMUpdate_(txpGroups, txpGroupCombinedWeights, origCounts, 
                  alphas, alphasPrime);
      }
    }

    // Truncate tiny expression values
    double alphaSum = 0.0;
    if (useVBEM and !perTranscriptPrior) {
      std::vector<double> cutoffs(transcripts.size(), 0.0);
      for (size_t i = 0; i < transcripts.size(); ++i) {
        cutoffs[i] = minAlpha;
      }
      // alphaSum = truncateCountVector(alphas, cutoffs);
      alphaSum = truncateCountVector(alphas, cutoffs);
    } else {
      // Truncate tiny expression values
      alphaSum = truncateCountVector(alphas, cutoff);
    }

    if (alphaSum < ::minWeight) {
      jointLog->error("Total alpha weight was too small! "
                      "Make sure you ran salmon correctly.");
      return false;
    }

    if (useScaledCounts) {
      double mappedFragsDouble = static_cast<double>(numMappedFrags);
      double alphaSum = 0.0;
      for (auto a : alphas) {
        alphaSum += a;
      }
      if (alphaSum > ::minWeight) {
        double scaleFrac = 1.0 / alphaSum;
        // scaleFrac converts alpha to nucleotide fraction,
        // and multiplying by numMappedFrags scales by the total
        // number of mapped fragments to provide an estimated count.
        for (auto& a : alphas) {
          a = mappedFragsDouble * (a * scaleFrac);
        }
      } else { // This shouldn't happen!
        sopt.jointLog->error(
            "Bootstrap had insufficient number of fragments!"
            "Something is probably wrong; please check that you "
            "have run salmon correctly and report this to GitHub.");
      }
    }

    writeBootstrap(alphas);
  }
  return true;
}

template <typename ExpT>
bool CollapsedEMOptimizer::gatherBootstraps(
    ExpT& readExp, SalmonOpts& sopt,
    std::function<bool(const std::vector<double>&)>& writeBootstrap,
    double relDiffTolerance, uint32_t maxIter) {


  std::vector<Transcript>& transcripts = readExp.transcripts();
  std::vector<bool> available(transcripts.size(), false);
  using VecT = CollapsedEMOptimizer::SerialVecType;
  // With atomics
  VecT alphas(transcripts.size(), 0.0);
  VecT alphasPrime(transcripts.size(), 0.0);
  VecT expTheta(transcripts.size());
  Eigen::VectorXd effLens(transcripts.size());
  double minAlpha = 1e-8;

  bool scaleCounts = (!sopt.useQuasi and !sopt.allowOrphans);

  uint64_t numMappedFrags =
      scaleCounts ? readExp.upperBoundHits() : readExp.numMappedFragments();

  uint32_t numBootstraps = sopt.numBootstraps;

  auto& eqBuilder = readExp.equivalenceClassBuilder();
  auto& eqVec = eqBuilder.eqVec();

  std::unordered_set<uint32_t> activeTranscriptIDs;
  const size_t numClasses = eqVec.size();
  for (size_t cid = 0; cid < numClasses; ++cid) { 
    auto nt = eqBuilder.getNumTranscriptsForClass(cid);
    auto& txps = eqVec[cid].first.txps;
    for (size_t tctr = 0; tctr < nt; ++tctr) {
      auto t = txps[tctr];
      transcripts[t].setActive();
      activeTranscriptIDs.insert(t);
    }
  }

  bool perTranscriptPrior{sopt.perTranscriptPrior};
  double priorValue{sopt.vbPrior};

  auto jointLog = sopt.jointLog;

  jointLog->info("Will draw {:n} bootstrap samples", numBootstraps);
  jointLog->info("Optimizing over {:n} equivalence classes", eqVec.size());

  double totalNumFrags{static_cast<double>(numMappedFrags)};
  double totalLen{0.0};

  if (activeTranscriptIDs.size() == 0) {
    jointLog->error("It seems that no transcripts are expressed; something is "
                    "likely wrong!");
    std::exit(1);
  }

  double scale = 1.0 / activeTranscriptIDs.size();
  for (size_t i = 0; i < transcripts.size(); ++i) {
    // double m = transcripts[i].mass(false);
    alphas[i] = transcripts[i].getActive() ? scale * totalNumFrags : 0.0;
    effLens(i) = (sopt.noEffectiveLengthCorrection)
                     ? transcripts[i].RefLength
                     : transcripts[i].EffectiveLength;
    totalLen += effLens(i);
  }

  // If we use VBEM, we'll need the prior parameters
  std::vector<double> priorAlphas = populatePriorAlphas_(
      transcripts, effLens, priorValue, perTranscriptPrior);

  auto numRemoved =
      markDegenerateClasses(eqVec, alphas, available, sopt.jointLog);
  sopt.jointLog->info("Marked {} weighted equivalence classes as degenerate",
                      numRemoved);

  // Since we will use the same weights and transcript groups for each
  // of the bootstrap samples (only the count vector will change), it
  // makes sense to keep only one copy of these.
  using TGroupLabelT = std::vector<uint32_t>;
  using TGroupWeightVec = std::vector<double>;
  std::vector<TGroupLabelT> txpGroups;
  std::vector<TGroupWeightVec> txpGroupCombinedWeights;
  std::vector<uint64_t> origCounts;
  uint64_t totalCount{0};

  for (size_t cid = 0; cid < numClasses; ++cid) { 
    const auto& kv = eqVec[cid];
    uint64_t count = kv.second.count;

    // for each transcript in this class
    const TranscriptGroup& tgroup = kv.first;
    if (tgroup.valid) {
      //const std::vector<uint32_t>& txps = tgroup.txps;
      const auto numTranscripts = eqBuilder.getNumTranscriptsForClass(cid);
      std::vector<uint32_t> txps(tgroup.txps.begin(), tgroup.txps.begin()+numTranscripts);
      const auto& auxs = kv.second.combinedWeights;
      
      if (txps.size() != auxs.size()) {
        sopt.jointLog->critical(
            "# of transcripts ({}) should match length of weight vec. ({})",
            txps.size(), auxs.size());
        sopt.jointLog->flush();
        spdlog::drop_all();
        std::exit(1);
      }
      
      txpGroups.push_back(txps);
      // Convert to non-atomic
      txpGroupCombinedWeights.emplace_back(auxs.begin(), auxs.end());
      origCounts.push_back(count);
      totalCount += count;
    }
  }

  double floatCount = totalCount;
  std::vector<double> samplingWeights(txpGroups.size(), 0.0);
  for (size_t i = 0; i < origCounts.size(); ++i) {
    samplingWeights[i] = origCounts[i] / floatCount;
  }

  size_t numWorkerThreads{1};
  if (sopt.numThreads > 1 and numBootstraps > 1) {
    numWorkerThreads = std::min(sopt.numThreads - 1, numBootstraps - 1);
  }

  std::atomic<uint32_t> bsCounter{0};
  std::vector<std::thread> workerThreads;
  for (size_t tn = 0; tn < numWorkerThreads; ++tn) {
    workerThreads.emplace_back(
        doBootstrap, std::ref(txpGroups), std::ref(txpGroupCombinedWeights),
        std::ref(transcripts), std::ref(effLens), std::ref(samplingWeights), std::ref(origCounts),
        totalCount, numMappedFrags, scale, std::ref(bsCounter), std::ref(sopt),
        std::ref(priorAlphas), std::ref(writeBootstrap), relDiffTolerance,
        maxIter);
  }

  for (auto& t : workerThreads) {
    t.join();
  }
  return true;
}

template <typename EQVecT>
void updateEqClassWeights(
    EQVecT& eqVec,
    Eigen::VectorXd& effLens) {



  tbb::parallel_for(
      BlockedIndexRange(size_t(0), size_t(eqVec.size())),
      [&eqVec, &effLens](const BlockedIndexRange& range) -> void {
        // For each index in the equivalence class vector
        for (auto eqID : boost::irange(range.begin(), range.end())) {
          // The vector entry
          auto& kv = eqVec[eqID];
          // The label of the equivalence class
          const TranscriptGroup& k = kv.first;
          // The size of the label
          size_t classSize = kv.second.weights.size(); // k.txps.size();
          // The weights of the label
          auto& v = kv.second;

          // Iterate over each weight and set it equal to
          // 1 / effLen of the corresponding transcript
          double wsum{0.0};
          for (size_t i = 0; i < classSize; ++i) {
            auto tid = k.txps[i];
            auto probStartPos = 1.0 / effLens(tid);
            v.combinedWeights[i] =
                kv.second.count * (v.weights[i] * probStartPos);
            wsum += v.combinedWeights[i];
          }
          double wnorm = 1.0 / wsum;
          for (size_t i = 0; i < classSize; ++i) {
            v.combinedWeights[i] *= wnorm;
          }
        }
      });
}

void CollapsedEMOptimizer::initAlpha(std::vector<Transcript>& transcripts,
                                     std::vector<std::atomic<double>>& alphas,
                                     std::vector<std::atomic<double>>& alphasPrime,
                                     std::vector<std::atomic<double>>& expTheta,
                                     double& totalWeight,
                                     Eigen::VectorXd& effLens,
                                     bool& useEffectiveLengths,
                                     SalmonOpts& sopt,
                                     int64_t& numActive,
                                     double& totalLen,
                                     bool& metaGenomeMode,
                                     bool& altInitMode)
{
    /// @brief 這個for在決定offline初值
    for (size_t i = 0; i < transcripts.size(); ++i) {
        auto& txp = transcripts[i];
        /// @brief 在normalizeAlpha那邊處理完projectedCounts了
        alphas[i] = txp.projectedCounts;

        /// @brief totalWeight是所有transcript的projectedCounts總和
        totalWeight += alphas[i];
        effLens(i) = useEffectiveLengths
                        ? std::exp(txp.getCachedLogEffectiveLength())
                        : txp.RefLength;
        if (sopt.noLengthCorrection) {
        effLens(i) = 100.0;
        }
        txp.EffectiveLength = effLens(i);

        double uniqueCount = static_cast<double>(txp.uniqueCount() + 0.5);
        /// @brief initUniform會用在offline的初始化, wi是offline abundance初值
        /// @brief 若effLens平均2500bp的話, wi約等於2.5*uniqueCount
        auto wi = (sopt.initUniform) ? 100.0 : (uniqueCount * 1e-3 * effLens(i)); 
        alphasPrime[i] = wi;
        ++numActive;
        totalLen += effLens(i);
    }


      // Based on the number of observed reads, use
  // a linear combination of the online estimates
  // and the uniform distribution.
  /// @brief totalWeigh=sum of alphas初始值是projectedCounts
  /// @brief numActive=transcriptome.size()
  double uniformPrior = totalWeight / static_cast<double>(numActive);
  double maxFrac = 0.999;
  /// @brief 我模擬的數量(20M, 30M)通常應該是後者吧
  /// @param numRequiredFragments : default 5千萬
  double fracObserved = std::min(maxFrac, totalWeight / sopt.numRequiredFragments);
// std::cerr << "totalWeight:" << totalWeight << std::endl;
  // Above, we placed the uniformative (uniform) initalization into the
  // alphasPrime variables.  If that's what the user requested, then copy those
  // over to the alphas
  if (sopt.initUniform) {
    for (size_t i = 0; i < alphas.size(); ++i) {
      alphas[i].store(alphasPrime[i].load());
      alphasPrime[i] = 1.0;
    }
  } else { // otherwise, initialize with a linear combination of the true and
           // uniform alphas
    for (size_t i = 0; i < alphas.size(); ++i) {
      /// @brief altInitMode表weigh unique mapping more while initializing, default false
      auto uniAbund = (metaGenomeMode or altInitMode) ? alphasPrime[i].load()
                                                      : uniformPrior; /// 所有transcript的projectedCounts總和/transcript.size()
      /// @brief fracObserved比例的projectedCount, 配上(1-fracObserved)比例的uniqueCount
      /// @brief e.g., 3千萬fragments的話 : 60%*projectedCounts[i] + 40%*projectedCounts總和/transcripts.size() 
      /// @brief (總之就是另外四成 uniqueCounts[i]或projectedCounts[:]/transcripts.size())
      alphas[i] =
          (alphas[i] * fracObserved) + (uniAbund * (1.0 - fracObserved));
// if (i == 11821) std::cerr << "newalphas[FBtr0086273]:" << alphas[i] << std::endl;
// if (i == 11822) std::cerr << "newalphas[FBtr0086274]:" << alphas[i] << std::endl;
      alphasPrime[i] = 1.0;
    }
  }
}

template <typename EQVecT>
void CollapsedEMOptimizer::computeCombinedWeights(EQVecT& eqVec, 
                                                  Eigen::VectorXd& effLens,
                                                  bool& noRichEq,
                                                  SalmonOpts& sopt)
{
    // If the user requested *not* to use "rich" equivalence classes,
  // then wipe out all of the weight information here and simply replace
  // the weights with the effective length terms (here, the *inverse* of
  // the effective length).  Otherwise, multiply the existing weight terms
  // by the effective length term.
  /// @brief eqVec是一個equivalence class
  tbb::parallel_for(
      BlockedIndexRange(size_t(0), size_t(eqVec.size())),
      [&eqVec, &effLens, noRichEq, &sopt](const BlockedIndexRange& range) -> void {
        // For each index in the equivalence class vector
for (auto eqID : boost::irange(range.begin(), range.end())) {
          // The vector entry
          auto& kv = eqVec[eqID];
          // The label of the equivalence class
          const TranscriptGroup& k = kv.first; /// @brief TranscriptGroup
          // The size of the label
          size_t classSize = kv.second.weights.size(); // k.txps.size();
          // The weights of the label
          auto& v = kv.second; /// @brief TranscriptValue

          // Iterate over each weight and set it
          double wsum{0.0};
/// @brief 這個transcriptGroup當中的transcript個數
for (size_t i = 0; i < classSize; ++i) {
            auto tid = k.txps[i];
            double el = effLens(tid);
            if (el <= 1.0) {
              el = 1.0;
            }
            if (noRichEq) {
              // Keep length factor separate for the time being
              v.weights[i] = 1.0;
            }
            // meaningful values.
            /// @brief probStartPos=1.0/el表示對於eqv class之中每個fragment count的期望起點個數, 
            /// @brief 算aln->logProb時1.0/ (l(t) – Ir(t) + 1) 是因爲fragment one by one的看, 
            /// @brief 而此處是整個eqv class的所有fragment一起看, 
            /// @brief 所以用effective length代替1.0 / (l(t) – Ir(t) + 1)
            auto probStartPos = 1.0 / el; 

            // combined weight
            /// @brief 用eqClassMode則eqClass內的Transcript都採同一個Pr(f_avg | ti)
            /// @brief weights[i] 就是auxProbs[i]
            /// @brief combinedWeights就是根據Pr(l)*Pr(a)*Pr(o)*Pr(p)
            /// @brief 1.0/el, 所以越長的話combinedWeights越低, 難怪被長的覆蓋的短transcript會false positive
            double wt = sopt.eqClassMode ? v.weights[i] : v.count * v.weights[i] * probStartPos;
// if (tid == 11821) std::cerr << "combinedWeights[FBtr0086273]_initialize:" << wt << std::endl;
// if (tid == 11822) std::cerr << "combinedWeights[FBtr0086274]_initialize:" << wt << std::endl;
            v.combinedWeights.push_back(wt);
            wsum += wt;
}

          double wnorm = 1.0 / wsum;
          for (size_t i = 0; i < classSize; ++i) {
            v.combinedWeights[i] = v.combinedWeights[i] * wnorm; /// @brief normalize到0~1
          }
} /// @brief for eqID in eqVec
      });
}

template <typename ExpT>
bool CollapsedEMOptimizer::optimize(ExpT& readExp, SalmonOpts& sopt,
                                    double relDiffTolerance, uint32_t maxIter) { ///結尾在:1079

  // <-- deprecated in TBB --> tbb::task_scheduler_init tbbScheduler(sopt.numThreads);
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, sopt.numThreads);
  std::vector<Transcript>& transcripts = readExp.transcripts();
std::vector<double> multimappedFrac(transcripts.size(), 1.0);
for (size_t i(0); i<transcripts.size(); ++i)
{
    std::cerr << transcripts[i].multimappedCount() << std::endl;
    if (transcripts[i].multimappedCount() == 0)
        multimappedFrac[i] = 1.0;
    else
        multimappedFrac[i] = (double)transcripts[i].totalCount() / (double)transcripts[i].multimappedCount();
}
  std::vector<bool> available(transcripts.size(), false);

  // An EM termination criterion, adopted from Bray et al. 2016
  uint32_t minIter = 50;
  bool seqBiasCorrect = sopt.biasCorrect;
  bool gcBiasCorrect = sopt.gcBiasCorrect;
  bool posBiasCorrect = sopt.posBiasCorrect;
  bool doBiasCorrect = seqBiasCorrect or gcBiasCorrect or posBiasCorrect;
  bool metaGenomeMode = sopt.meta;
  /// @brief Weigh unique reads more heavily when initialzing the optimization.
  bool altInitMode = sopt.alternativeInitMode;

  using VecT = CollapsedEMOptimizer::VecType;
  // With atomics
  VecType alphas(transcripts.size());
  VecType alphasPrime(transcripts.size());
  VecType expTheta(transcripts.size());

  Eigen::VectorXd effLens(transcripts.size());

  /// @brief countVec_ (i.e., vector<pair<TranscriptGroup, TranscriptValue>>, i.e., equivalence class個數)
  auto& eqVec =
      readExp.equivalenceClassBuilder().eqVec();
// std::cerr << "(TranscriptGroup.size()) eqVec().size()=" << eqVec.size() << std::endl;

  bool noRichEq = sopt.noRichEqClasses;

  bool useVBEM{sopt.useVBOpt};
  bool perTranscriptPrior{sopt.perTranscriptPrior};
  double priorValue{sopt.vbPrior}; /// @brief default: 0.01

  auto jointLog = sopt.jointLog;

  auto& fragStartDists = readExp.fragmentStartPositionDistributions();
  double totalNumFrags{static_cast<double>(readExp.numMappedFragments())};
  double totalLen{0.0};

  // If effective length correction isn't turned off, then use effective
  // lengths rather than reference lengths.
  bool useEffectiveLengths = !sopt.noEffectiveLengthCorrection;

  int64_t numActive{0};
  double totalWeight{0.0};

  initAlpha(transcripts, alphas, alphasPrime, expTheta, 
            totalWeight, effLens, useEffectiveLengths, sopt, numActive, totalLen, 
            metaGenomeMode, altInitMode);

  // If we use VBEM, we'll need the prior parameters
  std::vector<double> priorAlphas = populatePriorAlphas_(
      transcripts, effLens, priorValue, perTranscriptPrior);

  computeCombinedWeights(eqVec, effLens, noRichEq, sopt);

  /// @brief abundance*aux太小(1e-308)的話, 丟掉這個eqv class
  auto numRemoved =
      markDegenerateClasses(eqVec, alphas, available, sopt.jointLog);
  sopt.jointLog->info("Marked {} weighted equivalence classes as degenerate",
                      numRemoved);

  size_t itNum{0};

  // EM termination criteria, adopted from Bray et al. 2016
  double minAlpha = 1e-8;
  double alphaCheckCutoff = 1e-2; /// @brief transcriptome當中最大alpha變化量小於0.01
  double cutoff = minAlpha;

  // Iterations in which we will allow re-computing the effective lengths
  // if bias-correction is enabled.
  // std::vector<uint32_t> recomputeIt{100, 500, 1000};
  minIter = 100;

  bool converged{false};
  double maxRelDiff = -std::numeric_limits<double>::max();
  bool needBias = doBiasCorrect;
  size_t targetIt{10};
  /* -- v0.8.x
  double alphaSum = 0.0;
  */

/// @brief while (itNum < 100) or (itNum < 10000 and !converged) or (只有第一round會needBias)
while (itNum < minIter or (itNum < maxIter and !converged) or needBias) {
    if (needBias and (itNum > targetIt or converged)) {

      jointLog->info(
          "iteration {:n}, adjusting effective lengths to account for biases",
          itNum);
      effLens = salmon::utils::updateEffectiveLengths(sopt, readExp, effLens,
                                                      alphas, available, true);
      // if we're doing the VB optimization, update the priors
      if (useVBEM) {
        priorAlphas = populatePriorAlphas_(transcripts, effLens, priorValue,
                                           perTranscriptPrior);
      }

      // Check for strangeness with the lengths.
      for (int32_t i = 0; i < effLens.size(); ++i) {
        if (effLens(i) <= 0.0) {
          jointLog->warn("Transcript {} had length {}", i, effLens(i));
        }
      }
      updateEqClassWeights(eqVec, effLens);
      needBias = false;

      if ( sopt.eqClassMode ) {
        /// @brief 這句重要, Eqclass Model跟bias correction不會同時出現
        jointLog->error("Eqclass Mode should not be performing bias correction");
        jointLog->flush();
        exit(1);
      }
    } /// @brief if (needBias and (itNum > targetIt or converged)) 

    if (useVBEM) {
      VBEMUpdate_(eqVec, priorAlphas, alphas,
                  alphasPrime, expTheta, multimappedFrac);
    } else {
      /*
      if (itNum > 0 and (itNum % 250 == 0)) {
        for (size_t i = 0; i < transcripts.size(); ++i) {
      	  if (alphas[i] < 1.0) { alphas[i] = 0.0; }
      	}
      }
      */

      /// @brief eqVec : vec<pair<TranscriptGroup, TranscriptValue>>
      /// @brief priorAlphas : 0.01
      /// @brief 上面修改alphas是一定比例projectedCount+一定比例uniqueCount
      /// @brief alphasPrime : 1.0
      EMUpdate_(eqVec, priorAlphas, alphas, alphasPrime, multimappedFrac);
    }

    /// @brief 這個for是論文equation 12
    converged = true;
    maxRelDiff = -std::numeric_limits<double>::max();
    for (size_t i = 0; i < transcripts.size(); ++i) {
      if (alphasPrime[i] > alphaCheckCutoff) {
        double relDiff = std::abs(alphas[i] - alphasPrime[i]) / alphasPrime[i];
        maxRelDiff = (relDiff > maxRelDiff) ? relDiff : maxRelDiff;
        if (relDiff > relDiffTolerance) {
          converged = false;
        }
      }
// if(i == 11821) std::cerr << "alphas[FBtr0086273] out there1: " << alphas[i].load() << std::endl;
// if(i == 11822) std::cerr << "alphas[FBtr0086274] out there1: " << alphas[i].load() << std::endl;
      alphas[i].store(alphasPrime[i].load());
// if(i == 11821) std::cerr << "alphas[FBtr0086273] out there2: " << alphas[i].load() << std::endl;
// if(i == 11822) std::cerr << "alphas[FBtr0086274] out there2: " << alphas[i].load() << std::endl;
// std::cerr << std::endl;
      alphasPrime[i].store(0.0);
    }

    /* -- v0.8.x
    if (converged and itNum > minIter and !needBias) {
      if (useVBEM and !perTranscriptPrior) {
        std::vector<double> cutoffs(transcripts.size(), 0.0);
        for (size_t i = 0; i < transcripts.size(); ++i) {
          cutoffs[i] = minAlpha;
        }
        alphaSum = truncateCountVector(alphas, cutoffs);
      } else {
        // Truncate tiny expression values
        alphaSum = truncateCountVector(alphas, cutoff);
      }
      if (useVBEM) {
        VBEMUpdate_(eqVec, priorAlphas, alphas,
    alphasPrime, expTheta); } else { EMUpdate_(eqVec, transcripts, alphas,
    alphasPrime);
      }
      for (size_t i = 0; i < transcripts.size(); ++i) {
        alphas[i] = alphasPrime[i];
        alphasPrime[i] = 0.0;
      }
    }
    */

    if (itNum % 100 == 0) {
      jointLog->info("iteration = {:n} | max rel diff. = {}", itNum, maxRelDiff);
    }

    ++itNum;
} /// @brief while (itNum < minIter or (itNum < maxIter and !converged) or needBias) {

/// @brief normalized alphas
// double alphaDenom = 0.0;
// for (size_t i = 0; i < alphas.size(); ++i)
//     alphaDenom += alphas[i];
// for (size_t i=0; i<alphas.size(); ++i)
//     alphas[i] = alphas[i] / alphaDenom;

/// @brief 測試normalized alphas是否合理, 因此多試一round VBEM
// VBEMUpdate_(eqVec, priorAlphas, alphas,
//             alphasPrime, expTheta);

/// @brief 贏者全拿, 但只少要拿大於一半
// double threshold = 0.6;
// winnerTakesAll(eqVec, alphas, alphasPrime, threshold);

/// @brief 每個eqclass根據各自entropy調整count數
// eqClassInnerEntropy(eqVec, alphas, alphasPrime);

/// @brief 放至alphas
// for (size_t i = 0; i < transcripts.size(); ++i) {
//     alphas[i].store(alphasPrime[i].load());
//     alphasPrime[i].store(0.0);
// }

  /* -- v0.8.x
  if (alphaSum < ::minWeight) {
    jointLog->error("Total alpha weight was too small! "
                    "Make sure you ran salmon correctly.");
    return false;
  }
  */

  // Reset the original bias correction options
  sopt.gcBiasCorrect = gcBiasCorrect;
  sopt.biasCorrect = seqBiasCorrect;

  jointLog->info("iteration = {:n} | max rel diff. = {}", itNum, maxRelDiff);

  double alphaSum = 0.0;
  if (useVBEM and !perTranscriptPrior) {
    std::vector<double> cutoffs(transcripts.size(), 0.0);
    for (size_t i = 0; i < transcripts.size(); ++i) {
      cutoffs[i] = minAlpha;
    }
    alphaSum = truncateCountVector(alphas, cutoffs);
  } else {
    // Truncate tiny expression values
    /// @brief cutoff : default 1e-8
    alphaSum = truncateCountVector(alphas, cutoff);
  }

  if (alphaSum < ::minWeight) {
    jointLog->error("Total alpha weight was too small! "
                    "Make sure you ran salmon correctly.");
    return false;
  }

  // Set the mass of each transcript using the
  // computed alphas.
  for (size_t i = 0; i < transcripts.size(); ++i) {
    // Set the mass to the normalized (after truncation)
    // relative abundance
    // If we changed the effective lengths, copy them over here
    if (doBiasCorrect) {
      transcripts[i].EffectiveLength = effLens(i);
    }
    transcripts[i].setSharedCount(alphas[i]);
    transcripts[i].setMass(alphas[i] / alphaSum);
  }
  return true;
} /// @brief optimize結束

using BulkReadExperimentT = ReadExperiment<EquivalenceClassBuilder<TGValue>>;
template <typename FragT>
using BulkAlnLibT = AlignmentLibrary<FragT, EquivalenceClassBuilder<TGValue>>;
using SCReadExperimentT = ReadExperiment<EquivalenceClassBuilder<SCTGValue>>;


template bool CollapsedEMOptimizer::optimize<BulkReadExperimentT>(
    BulkReadExperimentT& readExp, SalmonOpts& sopt, double relDiffTolerance,
    uint32_t maxIter);

template bool CollapsedEMOptimizer::optimize<BulkAlnLibT<UnpairedRead>>(
    BulkAlnLibT<UnpairedRead>& readExp, SalmonOpts& sopt,
    double relDiffTolerance, uint32_t maxIter);

template bool CollapsedEMOptimizer::optimize<BulkAlnLibT<ReadPair>>(
    BulkAlnLibT<ReadPair>& readExp, SalmonOpts& sopt,
    double relDiffTolerance, uint32_t maxIter);


template bool CollapsedEMOptimizer::gatherBootstraps<BulkReadExperimentT>(
    BulkReadExperimentT& readExp, SalmonOpts& sopt,
    std::function<bool(const std::vector<double>&)>& writeBootstrap,
    double relDiffTolerance, uint32_t maxIter);

template bool
CollapsedEMOptimizer::gatherBootstraps<BulkAlnLibT<UnpairedRead>>(
    BulkAlnLibT<UnpairedRead>& readExp, SalmonOpts& sopt,
    std::function<bool(const std::vector<double>&)>& writeBootstrap,
    double relDiffTolerance, uint32_t maxIter);

template bool
CollapsedEMOptimizer::gatherBootstraps<BulkAlnLibT<ReadPair>>(
    BulkAlnLibT<ReadPair>& readExp, SalmonOpts& sopt,
    std::function<bool(const std::vector<double>&)>& writeBootstrap,
    double relDiffTolerance, uint32_t maxIter);
// Unused / old
