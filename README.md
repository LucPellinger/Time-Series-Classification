# Time-Series-Classification
This project aimed at exploring different deep learning architectures for time series classification. It includes a thoroughly implemented experimentation and evaluation pipeline as well as three corresponding DL architectures for multivariate time series classification. 

## ⚠️WARNING⚠️
This repository is currently imcomplete due to code maintance and pipeline debugging. Set a Star for further updates.


## 🛠️ Tech Stack

### 📊 Data Preprocessing & Benchmarking
- **[aeon](https://github.com/aeon-toolkit/aeon)** - A comprehensive Python toolkit for learning from time series
  - **Purpose**: Provides access to standard time series classification benchmark datasets from the UCR/UEA archives. Used for pipeline validation, model testing, and ensuring proper implementation of classification algorithms. Includes built-in data loaders for various time series formats and preprocessing utilities. Essential for verifying that reimplemented models (TimeMIL, TodyNet) are correctly integrated and performing as expected on established benchmarks.

### 🔧 Architecture/DNN Optimization
- **[Optuna](https://github.com/optuna/optuna)** - Next-generation hyperparameter optimization framework
  - **Purpose**: Automates the hyperparameter tuning process using advanced sampling algorithms (TPE, CMA-ES) and pruning strategies. Enables efficient exploration of hyperparameter spaces for neural network architectures, including learning rates, layer sizes, dropout rates, and architecture-specific parameters. Supports distributed optimization and visualization of optimization history.

### 🧠 Deep Learning Framework
- **[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)** - Lightweight PyTorch wrapper for high-performance AI research
  - **Purpose**: Simplifies the training loop implementation while maintaining flexibility for research. Handles distributed training, mixed precision, checkpointing, and logging automatically. Reduces boilerplate code and enforces best practices for reproducible deep learning experiments, allowing focus on model architecture rather than engineering details.

### 🔍 Model Interpretability
- **[Captum](https://github.com/pytorch/captum)** - Unified model interpretability library for PyTorch
  - **Purpose**: Provides attribution algorithms to understand which input features contribute most to model predictions. Implements methods like Integrated Gradients, SHAP, GradCAM, and Layer Conductance for time series models. Critical for understanding temporal patterns and feature importance in classification decisions, ensuring model transparency and trustworthiness.

### 📈 Hypothesis Testing and Evaluation
- **[SciPy](https://github.com/scipy/scipy)** - Fundamental library for scientific computing in Python
  - **Purpose**: Provides comprehensive statistical testing capabilities including t-tests, ANOVA, Wilcoxon tests, and correlation analyses. Used for evaluating model performance metrics, computing confidence intervals, and performing traditional statistical comparisons between models. Essential for rigorous statistical validation of results.

- **[deep-significance](https://github.com/Kaleidophon/deep-significance)** - Specialized significance testing for deep neural networks
  - **Purpose**: Addresses the unique challenges of comparing deep learning models where traditional statistical tests fall short. Implements Deep Dominance testing for proper model comparison, Almost Stochastic Order (ASO) for robust performance assessment, and handles the high variance typical in neural network training. Ensures meaningful and reliable conclusions about model superiority.

### 🤖 Time Series Classification Models
- **[TimeMIL](https://github.com/xiwenc1/TimeMIL)** - Time-aware Multiple Instance Learning
  - **Purpose**: Implements a novel approach that treats time series as bags of temporal instances, capturing both local and global temporal patterns. Particularly effective for datasets with varying sequence lengths and irregular sampling rates. Excels at identifying discriminative temporal segments within multivariate time series.

- **[TodyNet](https://github.com/liuxz1011/TodyNet)** - Temporal Dynamic Graph Neural Network
  - **Purpose**: Constructs dynamic graphs from multivariate time series to capture complex temporal and inter-variable dependencies. Uses graph neural networks to model evolving relationships between variables over time. Particularly powerful for datasets where variable interactions change dynamically and are crucial for classification.

---

## 📚 References

### Time Series Classification Models

**TimeMIL**
```bibtex
@article{chen2024timemil,
  title={TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning},
  author={Chen, Xiwen and Qiu, Peijie and Zhu, Wenhui and Li, Huayu and Wang, Hao and Sotiras, Aristeidis and Wang, Yalin and Razi, Abolfazl},
  journal={arXiv preprint arXiv:2405.03140},
  year={2024}
}
```

**TodyNet**
```bibtex
@article{Liu_2024,
  title={TodyNet: Temporal dynamic graph neural network for multivariate time series classification},
  volume={677},
  ISSN={0020-0255},
  url={http://dx.doi.org/10.1016/j.ins.2024.120914},
  DOI={10.1016/j.ins.2024.120914},
  journal={Information Sciences},
  publisher={Elsevier BV},
  author={Liu, Huaiyuan and Yang, Donghua and Liu, Xianzhang and Chen, Xinglei and Liang, Zhiyu and Wang, Hongzhi and Cui, Yong and Gu, Jun},
  year={2024},
  month=aug,
  pages={120914}
}
```

### Tools and Libraries

**aeon**
```bibtex
@article{aeon24jmlr,
  author  = {Matthew Middlehurst and Ali Ismail-Fawaz and Antoine Guillaume and Christopher Holder and David Guijo-Rubio and Guzal Bulatova and Leonidas Tsaprounis and Lukasz Mentel and Martin Walter and Patrick Sch{{\"a}}fer and Anthony Bagnall},
  title   = {aeon: a Python Toolkit for Learning from Time Series},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {289},
  pages   = {1--10},
  url     = {http://jmlr.org/papers/v25/23-1444.html}
}
```

**Optuna**
```bibtex
@inproceedings{akiba2019optuna,
  title={{O}ptuna: A Next-Generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={The 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2623--2631},
  year={2019}
}
```

**PyTorch Lightning**
```bibtex
@misc{pytorchlightning2019,
  title={PyTorch Lightning},
  author={Falcon, William and {The PyTorch Lightning team}},
  year={2019},
  doi={10.5281/zenodo.3828935},
  url={https://www.pytorchlightning.ai},
  note={The lightweight PyTorch wrapper for high-performance AI research}
}
```

**Captum**
```bibtex
@misc{kokhlikyan2020captum,
  title={Captum: A unified and generic model interpretability library for PyTorch},
  author={Narine Kokhlikyan and Vivek Miglani and Miguel Martin and Edward Wang and Bilal Alsallakh and Jonathan Reynolds and Alexander Melnikov and Natalia Kliushkina and Carlos Araya and Siqi Yan and Orion Reblitz-Richardson},
  year={2020},
  eprint={2009.07896},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

**SciPy**
```bibtex
@ARTICLE{2020SciPy-NMeth,
  author  = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
            Haberland, Matt and Reddy, Tyler and Cournapeau, David and
            Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
            Bright, Jonathan and {van der Walt}, St{\'e}fan J. and
            Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
            Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
            Kern, Robert and Larson, Eric and Carey, C J and
            Polat, {\.I}lhan and Feng, Yu and Moore, Eric W. and
            {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
            Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
            Harris, Charles R. and Archibald, Anne M. and
            Ribeiro, Ant{\^o}nio H. and Pedregosa, Fabian and
            {van Mulbregt}, Paul and {SciPy 1.0 Contributors}},
  title   = {{{SciPy} 1.0: Fundamental Algorithms for Scientific
            Computing in Python}},
  journal = {Nature Methods},
  year    = {2020},
  volume  = {17},
  pages   = {261--272},
  url     = {https://doi.org/10.1038/s41592-019-0686-2},
  doi     = {10.1038/s41592-019-0686-2}
}
```

**deep-significance**
```bibtex
@inproceedings{ulmer2022deep,
  title={deep-significance: Easy and Meaningful Significance Testing in the Age of Neural Networks},
  author={Ulmer, Dennis and Hardmeier, Christian and Frellsen, Jes},
  booktitle={ML Evaluation Standards Workshop at the Tenth International Conference on Learning Representations},
  year={2022}
}

@inproceedings{dror2019deep,
  author    = {Rotem Dror and Segev Shlomov and Roi Reichart},
  editor    = {Anna Korhonen and David R. Traum and Llu{\'{\i}}s M{\`{a}}rquez},
  title     = {Deep Dominance - How to Properly Compare Deep Neural Models},
  booktitle = {Proceedings of the 57th Conference of the Association for Computational
               Linguistics, {ACL} 2019, Florence, Italy, July 28- August 2, 2019,
               Volume 1: Long Papers},
  pages     = {2773--2785},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  url       = {https://doi.org/10.18653/v1/p19-1266},
  doi       = {10.18653/v1/p19-1266}
}

@incollection{del2018optimal,
  title={An optimal transportation approach for assessing almost stochastic order},
  author={Del Barrio, Eustasio and Cuesta-Albertos, Juan A and Matr{\'a}n, Carlos},
  booktitle={The Mathematics of the Uncertain},
  pages={33--44},
  year={2018},
  publisher={Springer}
}
```
