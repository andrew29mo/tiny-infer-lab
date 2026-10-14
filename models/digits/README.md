# Digits classifier artifact

`model.json` contains a trained 64→32→10 ReLU MLP, row-major float32 weights,
class mapping, exact train/test indices, training configuration, dataset hash,
and the original sklearn model's held-out probabilities for export verification.
It contains JSON numbers, not executable pickle/joblib objects. The project-trained
weights are distributed under the project's MIT license.

Dataset: E. Alpaydin and C. Kaynak (1998), **Optical Recognition of Handwritten
Digits**, UCI Machine Learning Repository, DOI
[10.24432/C50P49](https://doi.org/10.24432/C50P49).
The [UCI dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The dataset is supplied by the separately installed scikit-learn package; raw
images are not copied into this repository. Derived split indices, labels and
reference predictions are retained as evidence, with this attribution. MIT does
not replace the dataset's terms.

[scikit-learn load_digits](https://scikit-learn.org/1.6/modules/generated/sklearn.datasets.load_digits.html)
contains 1,797 8×8 digit images from the UCI test partition. This experiment makes
its own stratified random 1,347/450 train/test split of that subset; it does **not**
use the original UCI training/test protocol. Writer identity is unavailable here,
so no claim of writer-independent accuracy is made.

Preprocessing is fixed division by 16, with no fitted statistics. Seed 20261014,
architecture, optimizer and accuracy acceptance floor (94%) were fixed before
held-out evaluation. Training stopped by training-loss convergence (no test-set
selection or hyperparameter search). The iteration budget was increased from
500 to 1,500 after training-only nonconvergence, before examining test accuracy.
Retraining is explicit; ordinary tests use the committed model. Cross-platform
training can differ numerically, so exact weight reproducibility is not promised.

See [the real-model protocol and results](../../docs/REAL_MODEL.md).
