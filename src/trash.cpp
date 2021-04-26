/**
>HEADER
    Copyright (c) 2013, 2014, 2015, 2016 Rob Patro rob.patro@cs.stonybrook.edu

    This file is part of Salmon.

    Salmon is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Salmon is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Salmon.  If not, see <http://www.gnu.org/licenses/>.
<HEADER
**/
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// C++ string formatting library
#include "spdlog/fmt/fmt.h"

// C Includes for BWA
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

// Boost Includes
#include <boost/container/flat_map.hpp>
#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <boost/filesystem.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/program_options.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/thread/thread.hpp>

// Future C++ convenience classes
#include "core/range.hpp"

// TBB Includes
#include "tbb/blocked_range.h"
#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_unordered_set.h"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_for_each.h"
#include "tbb/parallel_reduce.h"
#include "tbb/partitioner.h"

// logger includes
#include "spdlog/spdlog.h"

// Cereal includes
#include "cereal/archives/binary.hpp"
#include "cereal/types/vector.hpp"

#include "concurrentqueue.h"

// salmon includes
#include "ClusterForest.hpp"
#include "FastxParser.hpp"
#include "IOUtils.hpp"
#include "LibraryFormat.hpp"
#include "ReadLibrary.hpp"
#include "SalmonConfig.hpp"
#include "SalmonDefaults.hpp"
#include "SalmonExceptions.hpp"
#include "SalmonIndex.hpp"
#include "SalmonMath.hpp"
#include "SalmonUtils.hpp"
#include "Transcript.hpp"
#include "SalmonMappingUtils.hpp"

#include "AlignmentGroup.hpp"
#include "BiasParams.hpp"
#include "CollapsedEMOptimizer.hpp"
#include "CollapsedGibbsSampler.hpp"
#include "EquivalenceClassBuilder.hpp"
#include "ForgettingMassCalculator.hpp"
#include "FragmentLengthDistribution.hpp"
#include "GZipWriter.hpp"

#include "EffectiveLengthStats.hpp"
#include "PairedAlignmentFormatter.hpp"
#include "ProgramOptionsGenerator.hpp"
#include "ReadExperiment.hpp"
//#include "RapMapUtils.hpp"
//#include "SACollector.hpp"
//#include "SASearcher.hpp"
//#include "HitManager.hpp"
#include "SalmonOpts.hpp"
//#include "SingleAlignmentFormatter.hpp"
#include "tsl/hopscotch_map.h"
#include "edlib.h"

#include "pufferfish/Util.hpp"
#include "pufferfish/MemCollector.hpp"
#include "pufferfish/MemChainer.hpp"
#include "pufferfish/SAMWriter.hpp"
#include "pufferfish/PuffAligner.hpp"
#include "pufferfish/ksw2pp/KSW2Aligner.hpp"
#include "pufferfish/metro/metrohash64.h"
#include "pufferfish/SelectiveAlignmentUtils.hpp"

/****** QUASI MAPPING DECLARATIONS *********/
using MateStatus = pufferfish::util::MateStatus;
using QuasiAlignment = pufferfish::util::QuasiAlignment;
using MergeResult = pufferfish::util::MergeResult;
/****** QUASI MAPPING DECLARATIONS  *******/

using paired_parser = fastx_parser::FastxParser<fastx_parser::ReadPair>;
using single_parser = fastx_parser::FastxParser<fastx_parser::ReadSeq>;

using TranscriptID = uint32_t;
using TranscriptIDVector = std::vector<TranscriptID>;
using KmerIDMap = std::vector<TranscriptIDVector>;

constexpr uint32_t miniBatchSize{5000};

template <typename AlnT> using AlnGroupVec = std::vector<AlignmentGroup<AlnT>>;

template <typename AlnT>
using AlnGroupVecRange = core::range<typename AlnGroupVec<AlnT>::iterator>;

#define __MOODYCAMEL__
#if defined(__MOODYCAMEL__)
template <typename AlnT>
using AlnGroupQueue = moodycamel::ConcurrentQueue<AlignmentGroup<AlnT>*>;
#else
template <typename AlnT>
using AlnGroupQueue = tbb::concurrent_queue<AlignmentGroup<AlnT>*>;
#endif

//#include "LightweightAlignmentDefs.hpp"

using ReadExperimentT = ReadExperiment<EquivalenceClassBuilder<TGValue>>;

template <typename AlnT>
void processMiniBatch(ReadExperimentT& readExp, ForgettingMassCalculator& fmCalc,
                      uint64_t firstTimestepOfRound, ReadLibrary& readLib,
                      const SalmonOpts& salmonOpts,
                      AlnGroupVecRange<AlnT> batchHits,
                      std::vector<Transcript>& transcripts,
                      ClusterForest& clusterForest,
                      FragmentLengthDistribution& fragLengthDist,
                      BiasParams& observedBiasParams,
                      /**
                       * NOTE : test new el model in future
                       * EffectiveLengthStats& obsEffLens,
                       */
                      std::atomic<uint64_t>& numAssignedFragments,
                      std::default_random_engine& randEng, bool initialRound,
                      std::atomic<bool>& burnedIn, double& maxZeroFrac,
                      distribution_utils::LogCMFCache& logCMFCache) {
std::cerr << "__FUNCTION__ = " << __FUNCTION__ << std::endl;
  using salmon::math::LOG_0;
  using salmon::math::LOG_1;
  using salmon::math::LOG_EPSILON;
  using salmon::math::LOG_ONEHALF;
  using salmon::math::logAdd;
  using salmon::math::logSub;

  const uint64_t numBurninFrags = salmonOpts.numBurninFrags;

  auto& log = salmonOpts.jointLog;
  // auto log = spdlog::get("jointLog");
  size_t numTranscripts{transcripts.size()};
  size_t localNumAssignedFragments{0};
  size_t priorNumAssignedFragments{numAssignedFragments};
  std::uniform_real_distribution<> uni(
      0.0, 1.0 + std::numeric_limits<double>::min());
  std::vector<uint64_t> libTypeCounts(LibraryFormat::maxLibTypeID() + 1);
  std::vector<uint64_t> libTypeCountsPerFrag(LibraryFormat::maxLibTypeID() + 1); /// @brief console's hits per frag
  bool hasCompatibleMapping{false};
  uint64_t numCompatibleFragments{0};

  std::vector<FragmentStartPositionDistribution>& fragStartDists =
      readExp.fragmentStartPositionDistributions();
  //auto& biasModel = readExp.sequenceBiasModel();
  auto& observedGCMass = observedBiasParams.observedGCMass;
  auto& obsFwd = observedBiasParams.massFwd;
  auto& obsRC = observedBiasParams.massRC;
  auto& observedPosBiasFwd = observedBiasParams.posBiasFW;
  auto& observedPosBiasRC = observedBiasParams.posBiasRC;

  bool posBiasCorrect = salmonOpts.posBiasCorrect;
  bool gcBiasCorrect = salmonOpts.gcBiasCorrect;
  bool updateCounts = initialRound;
  double incompatPrior = salmonOpts.incompatPrior;
  bool useFragLengthDist{!salmonOpts.noFragLengthDist};
  bool noFragLenFactor{salmonOpts.noFragLenFactor};
  bool useRankEqClasses{salmonOpts.rankEqClasses};
  uint32_t rangeFactorization{salmonOpts.rangeFactorizationBins};
  bool noLengthCorrection{salmonOpts.noLengthCorrection};
  bool useAuxParams = ((localNumAssignedFragments + numAssignedFragments) >=
                       salmonOpts.numPreBurninFrags);

  bool singleEndLib = !readLib.isPairedEnd();
  bool modelSingleFragProb = !salmonOpts.noSingleFragProb;

  // If we're auto detecting the library type
  auto* detector = readLib.getDetector();
  bool autoDetect = (detector != nullptr) ? detector->isActive() : false;
  // If we haven't detected yet, nothing is incompatible
  if (autoDetect) {
    incompatPrior = salmon::math::LOG_1;
  }

  uint64_t zeroProbFrags{0};

  // EQClass
  auto& eqBuilder = readExp.equivalenceClassBuilder();

  // Build reverse map from transcriptID => hit id
  using HitID = uint32_t;

  double logForgettingMass{0.0};
  uint64_t currentMinibatchTimestep{0};

  // logForgettingMass and currentMinibatchTimestep are OUT parameters!
  fmCalc.getLogMassAndTimestep(logForgettingMass, currentMinibatchTimestep);

  double startingCumulativeMass =
      fmCalc.cumulativeLogMassAt(firstTimestepOfRound);

  auto isUnexpectedOrphan = [](AlnT& aln, LibraryFormat expectedLibFormat) -> bool {
    return (expectedLibFormat.type == ReadType::PAIRED_END and
            aln.mateStatus != MateStatus::PAIRED_END_PAIRED);
  };

  if (modelSingleFragProb) {
    logCMFCache.refresh(numAssignedFragments.load(), burnedIn.load());
  }

  const size_t maxCacheLen{salmonOpts.fragLenDistMax};
  // Caches to avoid fld updates _within_ the set of alignments of a fragment 
  auto fetchPMF = [&fragLengthDist](size_t l) -> double { return fragLengthDist.pmf(l); };
  auto fetchCMF = [&fragLengthDist](size_t l) -> double { return fragLengthDist.cmf(l); };
  distribution_utils::IndexedVersionedCache<double> pmfCache(maxCacheLen);
  distribution_utils::IndexedVersionedCache<double> cmfCache(maxCacheLen);

  int i{0};
  {
    // Iterate over each group of alignments (a group consists of all alignments
    // reported
    // for a single read).  Distribute the read's mass to the transcripts
    // where it potentially aligns.
for (auto& alnGroup : batchHits) {
      pmfCache.increment_generation();
      cmfCache.increment_generation();

      // If we had no alignments for this read, then skip it
      if (alnGroup.size() == 0) {
        continue;
      }
      LibraryFormat expectedLibraryFormat = readLib.format();
      std::fill(libTypeCountsPerFrag.begin(), libTypeCountsPerFrag.end(), 0);

      // We start out with probability 0
      double sumOfAlignProbs{LOG_0};

      // Record whether or not this read is unique to a single transcript.
      bool transcriptUnique{true};

      auto firstTranscriptID = alnGroup.alignments().front().transcriptID();
      std::unordered_set<size_t> observedTranscripts;

      std::vector<uint32_t> txpIDs;
      std::vector<double> auxProbs;
      double auxDenom = salmon::math::LOG_0;

      uint32_t numInGroup{0};
      uint32_t prevTxpID{0};

      hasCompatibleMapping = false;
      useAuxParams = ((localNumAssignedFragments + numAssignedFragments) >=
                      salmonOpts.numPreBurninFrags);
      // For each alignment of this read
for (auto& aln : alnGroup.alignments()) {
        bool considerCondProb{burnedIn or useAuxParams};

        auto transcriptID = aln.transcriptID();
        auto& transcript = transcripts[transcriptID];
        transcriptUnique =
            transcriptUnique and (transcriptID == firstTranscriptID);

        double refLength =
            transcript.RefLength > 0 ? transcript.RefLength : 1.0;
        double coverage = aln.estAlnProb();
        double logFragCov = (coverage > 0) ? std::log(coverage) : LOG_1;

        // The alignment probability is the product of a
        // transcript-level term (based on abundance and) an
        // alignment-level term.
        double logRefLength{salmon::math::LOG_0};

        if (noLengthCorrection) {
          logRefLength = 1.0;
        } else if (salmonOpts.noEffectiveLengthCorrection or !burnedIn) {
          logRefLength = std::log(static_cast<double>(transcript.RefLength));
        } else {
          logRefLength = transcript.getCachedLogEffectiveLength();
        }

        double transcriptLogCount = transcript.mass(initialRound);
        auto flen = aln.fragLength();
        // If we have a properly-paired read then use the "pedantic"
        // definition here.
        if (aln.mateStatus == MateStatus::PAIRED_END_PAIRED and
            aln.fwd != aln.mateIsFwd) {
          flen = aln.fragLengthPedantic(transcript.RefLength);
        }

        // If the transcript had a non-zero count (including pseudocount)
        if (std::abs(transcriptLogCount) != LOG_0) {

          // The probability of drawing a fragment of this length;
          double logFragProb = LOG_1;

          // if we are modeling fragment probabilities for single-end mappings
          // and this is either a single-end library or an orphan.
          if (modelSingleFragProb and useFragLengthDist and (singleEndLib or isUnexpectedOrphan(aln, expectedLibraryFormat))) {
            // get the probability for a fragment of "ambiguous" length --- i.e. only the maximum length is bounded but
            // the fragment length is not completely characterized.
            logFragProb = logCMFCache.getAmbigFragLengthProb(aln.fwd, aln.pos, aln.readLen, transcript.CompleteLength, burnedIn.load());
          } else if (isUnexpectedOrphan(aln, expectedLibraryFormat)) {
            // If we are expecting a paired-end library, and this is an orphan,
            // then logFragProb should be small
            logFragProb = LOG_EPSILON;
          }

          if (flen > 0.0 and useFragLengthDist and considerCondProb) {
            size_t fl = flen;
            double lenProb = pmfCache.get_or_update(fl, fetchPMF);

            if (burnedIn) {
              /* condition fragment length prob on txp length */
              size_t rlen = static_cast<size_t>(refLength);
              double refLengthCM = cmfCache.get_or_update(fl, fetchCMF);

              bool computeMass =
                  fl < refLength and !salmon::math::isLog0(refLengthCM);
              logFragProb = (computeMass) ? (lenProb - refLengthCM)
                                          : salmon::math::LOG_EPSILON;
              if (computeMass and refLengthCM < lenProb) {
                // Threading is hard!  It's possible that an update to the PMF
                // snuck in between when we asked to cache the CMF and when the
                // "burnedIn" variable was last seen as false.
                log->info("reference length = {}, CMF[refLen] = {}, fragLen = "
                          "{}, PMF[fragLen] = {}",
                          refLength, std::exp(refLengthCM), aln.fragLength(),
                          std::exp(lenProb));
              }
            } else if (useAuxParams) {
              logFragProb = lenProb;
            }
            // logFragProb = lenProb;
            // logFragProb =
            // fragLengthDist.pmf(static_cast<size_t>(aln.fragLength()));
          }

          // TESTING
          if (noFragLenFactor) {
            logFragProb = LOG_1;
          }

          if (autoDetect) {
            detector->addSample(aln.libFormat());
            if (detector->canGuess()) {
              detector->mostLikelyType(readLib.getFormat());
              expectedLibraryFormat = readLib.getFormat();
              incompatPrior = salmonOpts.incompatPrior;
              autoDetect = false;
            } else if (!detector->isActive()) {
              expectedLibraryFormat = readLib.getFormat();
              incompatPrior = salmonOpts.incompatPrior;
              autoDetect = false;
            }
          }

          // TODO: Maybe take the fragment length distribution into account
          // for single-end fragments?

          // The probability that the fragments align to the given strands in
          // the
          // given orientations.
          bool isCompat = salmon::utils::isCompatible(
              aln.libFormat(), expectedLibraryFormat,
              static_cast<int32_t>(aln.pos), aln.fwd, aln.mateStatus);
          double logAlignCompatProb = isCompat ? LOG_1 : incompatPrior;
          if (!isCompat and salmonOpts.ignoreIncompat) {
            aln.logProb = salmon::math::LOG_0;
            continue;
          }

          /*
          double logAlignCompatProb =
              (useReadCompat) ? (salmon::utils::logAlignFormatProb(
                        aln.libFormat(), expectedLibraryFormat,
                        static_cast<int32_t>(aln.pos), aln.fwd,
                                        aln.mateStatus,
          salmonOpts.incompatPrior)) : LOG_1;
          */
          /** New compat handling
          // True if the read is compatible with the
          // expected library type; false otherwise.
          bool compat = ignoreCompat;
          if (!compat) {
              if (aln.mateStatus ==
              MateStatus::PAIRED_END_PAIRED) {
                  compat = salmon::utils::compatibleHit(
                          expectedLibType, observedLibType);
              } else {
                  int32_t pos = static_cast<int32_t>(aln.pos);
                  compat = salmon::utils::compatibleHit(
                          expectedLibraryFormat, pos,
                          aln.fwd, aln.mateStatus);
              }
          }
          **/

          // Allow for a non-uniform fragment start position distribution

          double startPosProb{-logRefLength};
          if (aln.mateStatus == MateStatus::PAIRED_END_PAIRED and
              !noLengthCorrection) {
            startPosProb = (flen <= refLength) ? -std::log(refLength - flen + 1)
                                               : salmon::math::LOG_EPSILON;
            // NOTE : test new el model in future
            // if (flen <= refLength) { obsEffLens.addFragment(transcriptID,
            // (refLength - flen + 1), logForgettingMass); }
          }

          double fragStartLogNumerator{salmon::math::LOG_1};
          double fragStartLogDenominator{salmon::math::LOG_1};

          auto hitPos = aln.hitPos();

          // Increment the count of this type of read that we've seen
          ++libTypeCountsPerFrag[aln.libFormat().formatID()];
          //
          if (!hasCompatibleMapping and logAlignCompatProb == LOG_1) {
            hasCompatibleMapping = true;
          }

          // The total auxiliary probabilty is the product (sum in log-space) of
          // The start position probability
          // The fragment length probabilty
          // The mapping score (coverage) probability
          // The fragment compatibility probability
          // The bias probability
          double auxProb = logFragProb + logFragCov + logAlignCompatProb;

          aln.logProb = transcriptLogCount + auxProb + startPosProb;

          // If this alignment had a zero probability, then skip it
          if (std::abs(aln.logProb) == LOG_0) {
            continue;
          }

          sumOfAlignProbs = logAdd(sumOfAlignProbs, aln.logProb);

          if (updateCounts and observedTranscripts.find(transcriptID) ==
                                   observedTranscripts.end()) {
            transcripts[transcriptID].addTotalCount(1);
            observedTranscripts.insert(transcriptID);
          }
          // EQCLASS
          if (transcriptID < prevTxpID) {
            std::cerr << "[ERROR] Transcript IDs are not in sorted order; "
                         "please report this bug on GitHub!\n";
          }
          prevTxpID = transcriptID;
          txpIDs.push_back(transcriptID);
          auxProbs.push_back(auxProb);
          auxDenom = salmon::math::logAdd(auxDenom, auxProb);
        } /// @brief if (std::abs(transcriptLogCount) != LOG_0)
        else {
          aln.logProb = LOG_0;
        }
} /// @brief for (auto& aln : alnGroup.alignments())

      // If this fragment has a zero probability,
      // go to the next one
      if (sumOfAlignProbs == LOG_0) {
        ++zeroProbFrags;
        continue;
      } else { // otherwise, count it as assigned
        ++localNumAssignedFragments;
        if (hasCompatibleMapping) {
          ++numCompatibleFragments;
        }
      }

      // EQCLASS
      double auxProbSum{0.0};
      for (auto& p : auxProbs) {
        p = std::exp(p - auxDenom);
        auxProbSum += p;
      }

      auto eqSize = txpIDs.size();
      if (eqSize > 0) {
        if (useRankEqClasses and eqSize > 1) {
          std::vector<int> inds(eqSize);
          std::iota(inds.begin(), inds.end(), 0);
          // Get the indices in order by conditional probability
          std::sort(inds.begin(), inds.end(),
                    [&auxProbs](int i, int j) -> bool {
                      return auxProbs[i] < auxProbs[j];
                    });
          {
            decltype(txpIDs) txpIDsNew(txpIDs.size());
            decltype(auxProbs) auxProbsNew(auxProbs.size());
            for (size_t r = 0; r < eqSize; ++r) {
              auto ind = inds[r];
              txpIDsNew[r] = txpIDs[ind];
              auxProbsNew[r] = auxProbs[ind];
            }
            std::swap(txpIDsNew, txpIDs);
            std::swap(auxProbsNew, auxProbs);
          }
        }

        if (rangeFactorization > 0) {
          int32_t txpsSize = txpIDs.size();
          int32_t rangeCount = std::sqrt(txpsSize) + rangeFactorization;

          for (int32_t i = 0; i < txpsSize; i++) {
            int32_t rangeNumber = auxProbs[i] * rangeCount;
            txpIDs.push_back(rangeNumber);
          }
        }

        TranscriptGroup tg(txpIDs);
        eqBuilder.addGroup(std::move(tg), auxProbs);
      }

      // normalize the hits
      for (auto& aln : alnGroup.alignments()) {
        if (std::abs(aln.logProb) == LOG_0) {
          continue;
        }
        // Normalize the log-probability of this alignment
        aln.logProb -= sumOfAlignProbs;
        // Get the transcript referenced in this alignment
        auto transcriptID = aln.transcriptID();
        auto& transcript = transcripts[transcriptID];

        // Add the new mass to this transcript
        double newMass = logForgettingMass + aln.logProb;
        transcript.addMass(newMass);

        // Paired-end
        if (aln.libFormat().type == ReadType::PAIRED_END) {
          // TODO: Is this right for *all* library types?
          if (aln.fwd) {
            obsFwd = salmon::math::logAdd(obsFwd, aln.logProb);
          } else {
            obsRC = salmon::math::logAdd(obsRC, aln.logProb);
          }
        } else if (aln.libFormat().type == ReadType::SINGLE_END) {
          int32_t p = (aln.pos < 0) ? 0 : aln.pos;
          if (static_cast<uint32_t>(p) >= transcript.RefLength) {
            p = transcript.RefLength - 1;
          }
          // Single-end or orphan
          if (aln.libFormat().strandedness == ReadStrandedness::S) {
            obsFwd = salmon::math::logAdd(obsFwd, aln.logProb);
          } else {
            obsRC = salmon::math::logAdd(obsRC, aln.logProb);
          }
        }

        if (posBiasCorrect) {
          auto lengthClassIndex = transcript.lengthClassIndex();
          switch (aln.mateStatus) {
          case MateStatus::PAIRED_END_PAIRED: {
            // TODO: Handle the non opposite strand case
            if (aln.fwd != aln.mateIsFwd) {
              int32_t posFW = aln.fwd ? aln.pos : aln.matePos;
              int32_t posRC = aln.fwd ? aln.matePos : aln.pos;
              posFW = posFW < 0 ? 0 : posFW;
              posFW = posFW >= static_cast<int32_t>(transcript.RefLength) ?
                               static_cast<int32_t>(transcript.RefLength) - 1
                                                    : posFW;
              posRC = posRC < 0 ? 0 : posRC;
              posRC = posRC >= static_cast<int32_t>(transcript.RefLength) ?
                               static_cast<int32_t>(transcript.RefLength) - 1
                                                    : posRC;
              observedPosBiasFwd[lengthClassIndex].addMass(
                  posFW, transcript.RefLength, aln.logProb);
              observedPosBiasRC[lengthClassIndex].addMass(
                  posRC, transcript.RefLength, aln.logProb);
            }
          } break;
          case MateStatus::PAIRED_END_LEFT:
          case MateStatus::PAIRED_END_RIGHT:
          case MateStatus::SINGLE_END: {
            int32_t pos = aln.pos;
            pos = pos < 0 ? 0 : pos;
            pos = pos >= static_cast<int32_t>(transcript.RefLength) ?
                         static_cast<int32_t>(transcript.RefLength) - 1 : pos;
            if (aln.fwd) {
              observedPosBiasFwd[lengthClassIndex].addMass(
                  pos, transcript.RefLength, aln.logProb);
            } else {
              observedPosBiasRC[lengthClassIndex].addMass(
                  pos, transcript.RefLength, aln.logProb);
            }
          } break;
          default:
            break;
          }
        }

        if (gcBiasCorrect) {
          if (aln.libFormat().type == ReadType::PAIRED_END) {
            int32_t start = std::min(aln.pos, aln.matePos);
            int32_t stop = start + aln.fragLen - 1;
            // WITH CONTEXT
            if (start >= 0 and stop < static_cast<int32_t>(transcript.RefLength)) {
              bool valid{false};
              auto desc = transcript.gcDesc(start, stop, valid);
              if (valid) {
                observedGCMass.inc(desc, aln.logProb);
              }
            }
          } else if (expectedLibraryFormat.type == ReadType::SINGLE_END) {
            // Both expected and observed should be single end here
            // For single-end reads, simply assume that every fragment
            // has a length equal to the conditional mean (given the
            // current transcript's length).
            auto cmeans = readExp.condMeans();
            auto cmean =
                static_cast<int32_t>((transcript.RefLength >= cmeans.size())
                                         ? cmeans.back()
                                         : cmeans[transcript.RefLength]);
            int32_t start = aln.fwd ? aln.pos : std::max(0, aln.pos - cmean);
            int32_t stop = start + cmean;
            // WITH CONTEXT
            if (start >= 0 and stop < static_cast<int32_t>(transcript.RefLength)) {
              bool valid{false};
              auto desc = transcript.gcDesc(start, stop, valid);
              if (valid) {
                observedGCMass.inc(desc, aln.logProb);
              }
            }
          }
        }
        double r = uni(randEng);
        if (!burnedIn and r < std::exp(aln.logProb)) {

          // Old fragment length calc: double fragLength = aln.fragLength();
          auto fragLength = aln.fragLengthPedantic(transcript.RefLength);
          if (fragLength > 0) {
            fragLengthDist.addVal(fragLength, logForgettingMass);
          }

        }
      } // end normalize

      // update the single target transcript
      if (transcriptUnique) { /// @brief looked uniquely assigned
        if (updateCounts) {
          transcripts[firstTranscriptID].addUniqueCount(1);
        }
        clusterForest.updateCluster(firstTranscriptID, 1.0, logForgettingMass,
                                    updateCounts);
      } else { // or the appropriate clusters
        clusterForest.mergeClusters<AlnT>(alnGroup.alignments().begin(),
                                          alnGroup.alignments().end());
        clusterForest.updateCluster(
            alnGroup.alignments().front().transcriptID(), 1.0,
            logForgettingMass, updateCounts);
      }

      for(size_t i=0; i < libTypeCounts.size(); ++i) {
        libTypeCounts[i] += (libTypeCountsPerFrag[i] > 0);
      }
} // end read group
  }   // end timer

  if (zeroProbFrags > 0) {
    auto batchReads = batchHits.size();
    maxZeroFrac = std::max(
        maxZeroFrac, static_cast<double>(100.0 * zeroProbFrags) / batchReads);
  }

  numAssignedFragments += localNumAssignedFragments;
  if (numAssignedFragments >= numBurninFrags and !burnedIn) {
    // NOTE: only one thread should succeed here, and that
    // thread will set burnedIn to true.
    readExp.updateTranscriptLengthsAtomic(burnedIn);
    fragLengthDist.cacheCMF();
  }
  if (initialRound) {
    readLib.updateLibTypeCounts(libTypeCounts);
    readLib.updateCompatCounts(numCompatibleFragments);
  }
}