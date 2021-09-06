#EVAN	BROWN	SAT	SEPT	19
#SVC_GPSF(Gamma Proportional to Scaling Factor)
#improved SVC that allows gamma to be set proportional to invariant or auto scaling factor ('scale' and 'auto')

from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.svm._base import BaseLibSVM, BaseSVC
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
# mypy error: error: Module 'sklearn.svm' has no attribute '_libsvm'
# (and same for other imports)
#from . import _libsvm as libsvm  # type: ignore
#from .import _liblinear as liblinear  # type: ignore
#from . import _libsvm_sparse as libsvm_sparse  # type: ignore
#from ..base import BaseEstimator, ClassifierMixin
#from ..preprocessing import LabelEncoder
#from ..utils.multiclass import _ovr_decision_function
#from ..utils import check_array, check_random_state
#from ..utils import column_or_1d
#from ..utils import compute_class_weight
#from ..utils.extmath import safe_sparse_dot
#from ..utils.validation import check_is_fitted, _check_large_sparse
#from ..utils.validation import _num_samples
#from ..utils.validation import _check_sample_weight, check_consistent_length
#from ..utils.multiclass import check_classification_targets
#from ..exceptions import ConvergenceWarning
#from ..exceptions import NotFittedError


LIBSVM_IMPL = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']

class SVC_GPSF(SVC, BaseSVC, BaseLibSVM, BaseEstimator): #improved SVC that allows gamma to be set proportional to invariant or auto scaling factor ('scale' and 'auto')

	#overwrote init and added ratio parameter
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', ratio = 1, #IMPROVEMENT	MADE~~~
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False,
                 random_state=None):
            
            self.ratio = ratio
            super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state)

    #overwrote fit method and updated gammas value in it		
    def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) \
                or (n_samples, n_samples)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like of shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)

        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object

        Notes
        -----
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """

        rnd = check_random_state(self.random_state)

        sparse = sp.isspmatrix(X)
        if sparse and self.kernel == "precomputed":
            raise TypeError("Sparse precomputed kernels are not supported.")
        self._sparse = sparse and not callable(self.kernel)

        if hasattr(self, 'decision_function_shape'):
            if self.decision_function_shape not in ('ovr', 'ovo'):
                raise ValueError(
                    "decision_function_shape must be either 'ovr' or 'ovo', "
                    "got {self.decision_function_shape}."
                )

        if callable(self.kernel):
            check_consistent_length(X, y)
        else:
            X, y = self._validate_data(X, y, dtype=np.float64,
                                       order='C', accept_sparse='csr',
                                       accept_large_sparse=False)

        y = self._validate_targets(y)

        sample_weight = np.asarray([]
                                   if sample_weight is None
                                   else sample_weight, dtype=np.float64)
        solver_type = LIBSVM_IMPL.index(self._impl)

        # input validation
        n_samples = _num_samples(X)
        if solver_type != 2 and n_samples != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n" +
                             "X has %s samples, but y has %s." %
                             (n_samples, y.shape[0]))

        if self.kernel == "precomputed" and n_samples != X.shape[1]:
            raise ValueError("Precomputed matrix must be a square matrix."
                             " Input is a {}x{} matrix."
                             .format(X.shape[0], X.shape[1]))

        if sample_weight.shape[0] > 0 and sample_weight.shape[0] != n_samples:
            raise ValueError("sample_weight and X have incompatible shapes: "
                             "%r vs %r\n"
                             "Note: Sparse matrices cannot be indexed w/"
                             "boolean masks (use `indices=True` in CV)."
                              % (sample_weight.shape, X.shape))

        kernel = 'precomputed' if callable(self.kernel) else self.kernel

        if kernel == 'precomputed':
            # unused but needs to be a float for cython code that ignores
            # it anyway
            self._gamma = 0.
        elif isinstance(self.gamma, str):
            if self.gamma == 'scale':
                # var = E[X^2] - E[X]^2 if sparse
                X_var = ((X.multiply(X)).mean() - (X.mean()) ** 2
                         if sparse else X.var())
                self._gamma = 1.0 / (X.shape[1] * X_var) * self.ratio if X_var != 0 else 1.0 #IMPROVEMENT	MADE
            elif self.gamma == 'auto':
                self._gamma = 1.0 / X.shape[1] * self.ratio #IMPROVEMENT	MADE~~~~~~
            else:
                raise ValueError(
                    "When 'gamma' is a string, it should be either 'scale' or "
                    "'auto'. Got '{}' instead.".format(self.gamma)
                )
        else:
            self._gamma = self.gamma

        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:
            print('[LibSVM]', end='')

        seed = rnd.randint(np.iinfo('i').max)
        fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
        # see comment on the other call to np.iinfo in this file

        self.shape_fit_ = X.shape if hasattr(X, "shape") else (n_samples, )

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function. Use self._intercept_ and self._dual_coef_
        # internally.
        self._intercept_ = self.intercept_.copy()
        self._dual_coef_ = self.dual_coef_
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            self.intercept_ *= -1
            self.dual_coef_ = -self.dual_coef_

        return self

#EXAMPLE:~~~~~~
# from sklearn.datasets import load_digits
# X,y = load_digits(return_X_y=1)
# svc_improved = SVC_GPSF(ratio = 0.5 , gamma = 'scale')
# svc_improved.fit(X[:10],y[:10])
# svc_improved.predict(X[:2])
# print(svc_improved)