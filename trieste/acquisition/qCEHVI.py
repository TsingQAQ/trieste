import tensorflow as tf
from itertools import combinations
from typing import Union, Mapping
from trieste.utils.pareto import Pareto
from .function import get_reference_point, SingleModelBatchAcquisitionBuilder, BatchAcquisitionFunctionBuilder
from trieste.acquisition.function import DEFAULTS, Dataset, ProbabilisticModel, \
    AcquisitionFunction, TensorType, BatchReparametrizationSampler, BatchAcquisitionFunction
from trieste.acquisition.rule import OBJECTIVE, CONSTRAINT
from math import inf


class BatchMonteCarloHypervolumeExpectedImprovement(SingleModelBatchAcquisitionBuilder):
    """
    Use of the inclusion-exclusion method
    refer
    @article{daulton2020differentiable,
    title={Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization},
    author={Daulton, Samuel and Balandat, Maximilian and Bakshy, Eytan},
    journal={arXiv preprint arXiv:2006.05078},
    year={2020}
    }
    """

    def __init__(self, sample_size: int = 512, *, jitter: float = DEFAULTS.JITTER,
                 nadir_setting: Union[str, callable] = "default"):
        """
        :param sample_size: The number of samples for each batch of points.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter
        self.q = -1
        self._nadir_setting = nadir_setting

    def _calculate_nadir(self, pareto: Pareto, nadir_setting="default"):
        """
        calculate the reference point for hypervolme calculation
        :param pareto: Pareto class
        :param nadir_setting
        """
        if nadir_setting == "default":
            return get_reference_point(pareto.front)
        else:
            assert callable(nadir_setting), ValueError(
                "nadir_setting: {} do not understood".format(nadir_setting)
            )
            return nadir_setting(pareto.front)

    def __repr__(self) -> str:
        """"""
        return f"BatchMonteCarloExpectedImprovement({self._sample_size!r}, jitter={self._jitter!r})"

    def _cache_q_subset_indices(self, q: int) -> None:
        r"""Cache indices corresponding to all subsets of `q`.
        This means that consecutive calls to `forward` with the same
        `q` will not recompute the indices for all (2^q - 1) subsets.
        Note: this will use more memory than regenerating the indices
        for each i and then deleting them, but it will be faster for
        repeated evaluations (e.g. during optimization).
        Args:
            q: batch size
        """
        if q != self.q:
            indices = list(range(q))
            self.q_subset_indices = {
                f"q_choose_{i}": tf.constant(list(combinations(indices, i)))
                for i in range(1, q + 1)
            }
            self.q = q

    def prepare_acquisition_function(
            self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model over the specified ``dataset``. Must have event shape [1].
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """

        tf.debugging.assert_positive(len(dataset), message='Dataset must be populated.')

        means, _ = model.predict(dataset.query_points)

        datasets_mean = tf.concat(means, axis=1)
        _pf = Pareto(Dataset(query_points=tf.zeros_like(datasets_mean), observations=datasets_mean))
        _nadir_pt = self._calculate_nadir(_pf, nadir_setting=self._nadir_setting)
        lb_points, ub_points = _pf.get_partitioned_cell_bounds(_nadir_pt)
        sampler = BatchReparametrizationSampler(self._sample_size, model)

        def batch_hvei(at: TensorType) -> TensorType:
            """
            :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            Complexity: O(num_obj * SK(2^q - 1))
            """
            # [..., S, B, num_obj]
            samples = sampler.sample(at, jitter=self._jitter)

            q = at.shape[-2]  # B
            self._cache_q_subset_indices(q)

            areas_per_segment = None
            # Inclusion-Exclusion loop
            for j in range(1, q + 1):
                # choose combination
                q_choose_j = self.q_subset_indices[f"q_choose_{j}"]
                # get combination of subsets: [..., S, B, num_obj] -> [..., S, Cq_j, j, num_obj]
                obj_subsets = tf.gather(samples, q_choose_j, axis=-2)
                # get lower vertices of overlap: [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
                overlap_vertices = tf.reduce_max(obj_subsets, axis=-2)

                # compare overlap vertices and lower bound of each cell: -> [..., S, K, Cq_j, num_obj]
                overlap_vertices = tf.maximum(tf.expand_dims(overlap_vertices, -3),
                                              lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :])

                # get hvi length within each cell:-> [..., S, Cq_j, K, num_obj]
                lengths_j = tf.maximum((ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :]
                                        - overlap_vertices), 0.0)
                # take product over hyperrectangle side lengths to compute area within each K
                # sum over all subsets of size Cq_j #
                areas_j = tf.reduce_sum(tf.reduce_prod(lengths_j, axis=-1), axis=-1)
                # [..., S, K]
                areas_per_segment = (-1) ** (j + 1) * areas_j if areas_per_segment is None \
                    else areas_per_segment + (-1) ** (j + 1) * areas_j

            # sum over segments(cells) and average over MC samples
            # return tf.reduce_mean(batch_improvement, axis=-1, keepdims=True)  # [..., 1]
            areas_in_total = tf.reduce_sum(areas_per_segment, axis=-1)
            return tf.reduce_mean(areas_in_total, axis=-1, keepdims=True)

        return batch_hvei


# TODO: Only use this one
class BatchMonteCarloConstraintHypervolumeExpectedImprovement(BatchAcquisitionFunctionBuilder):
    """
    The qEHVI considering constraint case, mainly refer sec 3.4 and A.3
    """

    def __init__(self, sample_size: int = 512, *, jitter: float = DEFAULTS.JITTER, eta: float = 1e-3):
        """
        :param sample_size: The number of samples for each batch of points.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :param eta temperature parameter for soft transforming of constraint used for sigmoid function,
            refer Eq. 11 of A. 3
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter
        self.q = -1
        self.eta = eta

    def _cache_q_subset_indices(self, q: int) -> None:
        r"""Cache indices corresponding to all subsets of `q`.
        This means that consecutive calls to `forward` with the same
        `q` will not recompute the indices for all (2^q - 1) subsets.
        Note: this will use more memory than regenerating the indices
        for each i and then deleting them, but it will be faster for
        repeated evaluations (e.g. during optimization).
        Args:
            q: batch size
        """
        if q != self.q:
            indices = list(range(q))
            self.q_subset_indices = {
                f"q_choose_{i}": tf.constant(list(combinations(indices, i)))
                for i in range(1, q + 1)
            }
            self.q = q

    def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> BatchAcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: A batch acquisition function.
        """
        assert OBJECTIVE in datasets.keys() and OBJECTIVE in models.keys(), ValueError(
            f"dict of models and datasets must contain the key {OBJECTIVE}, got keys {models.keys()}"
            f"and {datasets.keys()}")
        assert CONSTRAINT in datasets.keys() and OBJECTIVE in models.keys(), ValueError(
            f"dict of models and datasets must contain the key {CONSTRAINT}, got keys {models.keys()}"
            f"and {datasets.keys()}")

        obj_model = models[OBJECTIVE]
        con_model = models[CONSTRAINT]
        obj_means, _ = obj_model.predict(datasets[OBJECTIVE].query_points)  # [..., num_obj]
        con_means, _ = con_model.predict(datasets[CONSTRAINT].query_points)  # [..., num_con]
        fea_idx = tf.squeeze(tf.where(tf.reduce_all(con_means < 0, axis=-1)))  # [...] <0 denotes feasible
        fea_datasets_mean = tf.gather(obj_means, fea_idx, axis=0)
        _pf = Pareto(fea_datasets_mean)
        ref_pt = get_reference_point(_pf.front)
        lb_points, ub_points = _pf.get_hyper_cell_bounds(tf.constant([-inf] * fea_datasets_mean.shape[-1],
                                                                     dtype=fea_datasets_mean.dtype), ref_pt)
        obj_sampler = BatchReparametrizationSampler(self._sample_size, obj_model)
        con_sampler = BatchReparametrizationSampler(self._sample_size, con_model)

        def batch_hvei(at: TensorType) -> TensorType:
            """
            :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            Complexity: O(num_obj * SK(2^q - 1))
            """
            tf.print(f'at{at}')
            tf.debugging.assert_all_finite(at, 'NaN detected')
            # [..., S, B, num_obj]
            obj_samples = obj_sampler.sample(at, jitter=self._jitter)
            # [..., S, B, num_con]
            con_samples = con_sampler.sample(at, jitter=self._jitter)

            q = at.shape[-2]  # B
            self._cache_q_subset_indices(q)

            areas_per_segment = None
            # Inclusion-Exclusion loop
            for j in range(1, q + 1):
                # choose combination
                q_choose_j = self.q_subset_indices[f"q_choose_{j}"]
                # get combination of subsets: [..., S, B, num_obj] -> [..., S, Cq_j, j, num_obj]
                obj_subsets = tf.gather(obj_samples, q_choose_j, axis=-2)
                # get lower vertices of overlap: [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
                overlap_vertices = tf.reduce_max(obj_subsets, axis=-2)

                # compare overlap vertices and lower bound of each cell: -> [..., S, K, Cq_j, num_obj]
                overlap_vertices = tf.maximum(tf.expand_dims(overlap_vertices, -3),
                                              lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :])

                # get hvi length within each cell:-> [..., S, Cq_j, K, num_obj]
                lengths_j = tf.maximum((ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :]
                                        - overlap_vertices), 0.0)
                # refer: Eq. 10 of A. 3: [..., S, B], note <0 denote feasible
                fea = tf.reduce_prod(tf.sigmoid(-con_samples/self.eta), axis=-1)
                # [..., S, Cq_j, j]
                fea = tf.gather(fea, q_choose_j, axis=-1)
                # [..., S, Cq_j]
                fea = tf.reduce_prod(fea, axis=-1)
                # take product over hyperrectangle side lengths to compute area within each K
                # areas_j = tf.reduce_sum(tf.reduce_prod(lengths_j, axis=-1), axis=-1)
                areas_j = tf.reduce_prod(lengths_j, axis=-1) * fea[:, :, tf.newaxis, :]
                # sum over all subsets of size Cq_j #
                areas_j = tf.reduce_sum(areas_j, axis=-1)
                # [..., S, K]
                areas_per_segment = (-1) ** (j + 1) * areas_j if areas_per_segment is None \
                    else areas_per_segment + (-1) ** (j + 1) * areas_j

            # sum over segments(cells) and average over MC samples
            # return tf.reduce_mean(batch_improvement, axis=-1, keepdims=True)  # [..., 1]
            areas_in_total = tf.reduce_sum(areas_per_segment, axis=-1)
            tf.debugging.assert_all_finite(tf.reduce_mean(areas_in_total, axis=-1, keepdims=True), 'NaN detected')
            return tf.reduce_mean(areas_in_total, axis=-1, keepdims=True)

        return batch_hvei
