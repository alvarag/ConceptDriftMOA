# ConceptDriftMOA
Machine Learning algorithms for MOA designed to cope with concept drift. The versions are not optimized and have some limitations, please review the header of each class.

The original MOA software is necessary: https://github.com/Waikato/moa

* IB3: Instance-Based Learning.
* TWF: Time-Weighted Forgetting.
* LWF: Locally-Weighted Forgetting.
* ANNCAD: Adaptive NN Classification Algorithm for data-streams.
* AES: Data stream classification with artificial endocrine system.
* PECS: Prediction Error Context Switching.
* oi-GRLVQ: Online and incremental GRLVQ. 

## Citation policy
The code was implemented by Álvar Arnaiz-González. 

### Cite this software as:
 **Gunn, I. A., Arnaiz-González, Á., & Kuncheva, L. I. (2018)**. _A Taxonomic Look at Instance-based Stream Classifiers_. Neurocomputing, 286, 167-168. [doi: 10.1016/j.neucom.2018.01.062](https://doi.org/10.1016/j.neucom.2018.01.062)

```
@article{GunnArnaizKuncheva2018,
  title = "A taxonomic look at instance-based stream classifiers",
  journal = "Neurocomputing",
  volume = "286",
  pages = "167 - 178",
  year = "2018",
  issn = "0925-2312",
  doi = "https://doi.org/10.1016/j.neucom.2018.01.062",
  url = "http://www.sciencedirect.com/science/article/pii/S092523121830095X",
  author = "Iain Gunn and \'Alvar Arnaiz-Gonz\'alez and Ludmila I. Kuncheva"
}
```

## Contributions
Some of the algorithms have been adapted to MOA by means a wrapper, the original codes are available here:

* SimC: https://www.dropbox.com/s/s2t2ogaki1x1n4w/Weka.rar?dl=0
* SyncStream: https://github.com/kramerlab/SyncStream/
* IBLStreams: https://www.uni-marburg.de/fb12/kebi/research/software/iblstreams?language_sync=1
