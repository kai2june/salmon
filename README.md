[![Build Status](https://travis-ci.org/COMBINE-lab/salmon.svg?branch=master)](https://travis-ci.org/COMBINE-lab/salmon)
[![Documentation Status](https://readthedocs.org/projects/salmon/badge/?version=latest)](http://salmon.readthedocs.org/en/latest)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square)](http://bioconda.github.io/recipes/salmon/README.html)


**Try out alevin (salmon's single-cell processing module)!  Get started with the [tutorial](https://combine-lab.github.io/alevin-tutorial/#blog)**

**Help guide the development of Salmon, [take our survey](https://docs.google.com/forms/d/e/1FAIpQLSeWhBNE_fA_0uVHvbAlAulDmfmowv7rAYla879DZpqCARyRTQ/viewform)**

### Pre-computed decoy transcriptomes 

Although the precomputed decoys (<=v.14.2) are still compatible with the latest major release (v1.0.0). We highly recommend updating your index using the full genome, as it gives significantly higher accuracy, for more information please check our extensive benchmarking on comparing different alignment methods and their performance on RNA-seq quantification in the latest revised preprint [manuscript](https://www.biorxiv.org/content/10.1101/657874v2). tl;dr: fast is good but fast and accurate is better !
Please use the [tutorial](https://combine-lab.github.io/alevin-tutorial/2019/selective-alignment/) for a step-by-step guide on to efficiently index the reference transcriptome and genome for accurate gentrome based RNA-seq quantification.

**Facing problems with Indexing ?, [Check if anyone else already had this problem in the issues section or fill the index generation request form](https://forms.gle/3baJc5SYrkSWb1z48)**

What is Salmon?
===============

Salmon is a **wicked**-fast program to produce a highly-accurate, transcript-level quantification estimates from 
RNA-seq data.  Salmon achieves its accuracy and speed via a number of different innovations, including the 
use of *selective-alignment* (accurate but fast-to-compute proxies for traditional read alignments), and 
massively-parallel stochastic collapsed variational inference.  The result is a versatile tool that fits nicely
into many different pipelines.  For example, you can choose to make use of our *selective-alignment* algorithm by providing Salmon with raw sequencing reads, or, if it is more convenient, you can provide Salmon with regular alignments (e.g. an **unsorted** BAM file with alignments to the transcriptome produced with your favorite aligner), and it will use the same **wicked**-fast, state-of-the-art inference algorithm to estimate transcript-level abundances for your experiment.

Give salmon a try!  You can find the latest binary releases [here](https://github.com/COMBINE-lab/salmon/releases).

The current version number of the master branch of Salmon can be found [**here**](http://combine-lab.github.io/salmon/version_info/latest)

**NOTE**: Salmon works by (quasi)-mapping sequencing reads directly to the *transcriptome*.  This means the Salmon index should be built on a set of target transcripts, **not** on the *genome* of the underlying organism.  If indexing appears to be taking a very long time, or using a tremendous amount of memory (which it should not), please ensure that you are not attempting to build an index on the genome of your organism!

Documentation
==============

The documentation for Salmon is available on [ReadTheDocs](http://readthedocs.org), check it out [here](http://salmon.readthedocs.org).

Salmon is, and will continue to be, [freely and actively supported on a best-effort basis](https://oceangenomics.com/about/#open).
If you need industrial-grade technical support, please consider the options at [oceangenomics.com/support](http://oceangenomics.com/support).

Chat live about Salmon
======================

You can chat with the Salmon developers and other users via Gitter!

[![Join the chat at https://gitter.im/COMBINE-lab/salmon](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/COMBINE-lab/salmon?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
