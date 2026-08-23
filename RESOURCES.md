# Resources

Every dataset and library the project relies on, numbered in order of first appearance in
the pipeline. The dataset is used under the license noted for each entry; the sample
images and histogram figures generated from its photos are derivative works.

1. **LV-MHP v1.0 (Multiple-Human Parsing dataset)** — J. Li, J. Zhao, Y. Wei, C. Lang,
   Y. Li, T. Sim, S. Yan, J. Feng, *Multiple-Human Parsing in the Wild*, arXiv:1705.07206,
   2017. Learning and Vision (LV) Group, National University of Singapore. 4,980
   multi-person photographs; only the photos are used here, as raw material for
   synthesising single- and double-compressed samples. **Made freely available for
   non-commercial purposes (academic research, teaching, scientific publication, personal
   experimentation) only — NOT cleared for commercial use.** The sample images
   (`assets/sample.jpg`, `assets/sample_double.jpg`) and the histogram figures derived
   from them are derivative works under the same terms; a commercial deployment must
   substitute a commercially-licensed image source.
   https://lv-mhp.github.io/

2. **PyTorch** — A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep
   Learning Library*, NeurIPS, 2019. BSD-3-Clause. Defines, trains and serves the MLP
   classifier.
   https://pytorch.org

3. **NumPy** — C. R. Harris et al., *Array programming with NumPy*, Nature, 2020.
   BSD-3-Clause. Array representation of images, blocks and features throughout.
   https://numpy.org

4. **SciPy** — P. Virtanen et al., *SciPy 1.0: Fundamental Algorithms for Scientific
   Computing in Python*, Nature Methods, 2020. BSD-3-Clause. Its `fftpack` DCT/IDCT
   implement the coefficient transforms.
   https://scipy.org

5. **scikit-learn** — F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*,
   JMLR, 2011. BSD-3-Clause. Train/validation/test splitting and the classification
   metrics.
   https://scikit-learn.org

6. **Pillow** — A. Clark and contributors (originally F. Lundh's PIL). MIT-CMU (HPND)
   license. Loads images and re-encodes them as JPEG to simulate compression.
   https://python-pillow.org

7. **Matplotlib** — J. D. Hunter, *Matplotlib: A 2D Graphics Environment*, Computing in
   Science & Engineering, 2007. Matplotlib license (PSF-derived). Renders the loss, ROC,
   confusion-matrix and histogram figures.
   https://matplotlib.org

8. **tqdm** — C. da Costa-Luis and tqdm collaborators. MPL-2.0 or MIT, at your option.
   Progress bars during feature extraction and training.
   https://tqdm.github.io

9. **colorama** — J. Hartley and contributors. BSD-3-Clause. Cross-platform coloured
   console output.
   https://github.com/tartley/colorama

Supporting libraries (transitive dependencies of the above, all permissively licensed):
contourpy (BSD-3-Clause), cycler (BSD-3-Clause), fonttools (MIT), kiwisolver
(BSD-3-Clause), narwhals (MIT), packaging (Apache-2.0 or BSD-2-Clause), pyparsing (MIT),
python-dateutil (Apache-2.0 or BSD-3-Clause), six (MIT), threadpoolctl (BSD-3-Clause).

All code libraries above are cleared for commercial use under their permissive licenses;
the only non-commercial item in this project is the LV-MHP v1.0 dataset (entry 1), whose
images must be replaced before any commercial publication of derived results.
