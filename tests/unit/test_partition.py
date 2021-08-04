from __future__ import annotations

import numpy.testing as npt
import tensorflow as tf
import pytest

from trieste.utils.mo_utils.partition import ExactPartition2dNonDominated, DividedAndConquerNonDominated


def test_exact_partition_2d_bounds() -> None:
    objectives = tf.constant(
        [
            [0.1576, 0.7922],
            [0.4854, 0.0357],
            [0.1419, 0.9340],
        ]
    )

    partition_2d = ExactPartition2dNonDominated(objectives)

    npt.assert_array_equal(
        partition_2d._bounds.lower_idx, tf.constant([[0, 0], [1, 0], [2, 0], [3, 0]])
    )
    npt.assert_array_equal(
        partition_2d._bounds.upper_idx, tf.constant([[1, 4], [2, 1], [3, 2], [4, 3]])
    )
    npt.assert_allclose(
        partition_2d.front, tf.constant([[0.1419, 0.9340], [0.1576, 0.7922], [0.4854, 0.0357]])
    )


def test_exact_partition_2d_raise_when_input_is_not_pareto_front():
    objectives = tf.constant(
        [
            [0.9575, 0.4218],
            [0.9649, 0.9157],
            [0.1576, 0.7922],
            [0.9706, 0.9595],
            [0.9572, 0.6557],
            [0.4854, 0.0357],
            [0.8003, 0.8491],
            [0.1419, 0.9340],
        ]
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        ExactPartition2dNonDominated(objectives)


def test_divide_conquer_non_dominated_three_dimension_case() -> None:
    objectives = tf.constant(
        [
            [0.0, 2.0, 1.0],
            [7.0, 6.0, 0.0],
            [9.0, 0.0, 1.0],
        ]
    )

    partition_nd = DividedAndConquerNonDominated(objectives)

    npt.assert_array_equal(
        partition_nd._bounds.lower_idx,
        tf.constant(
            [
                [3, 2, 0],
                [3, 1, 0],
                [2, 2, 0],
                [2, 1, 0],
                [3, 0, 1],
                [2, 0, 1],
                [2, 0, 0],
                [0, 1, 1],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    )
    npt.assert_array_equal(
        partition_nd._bounds.upper_idx,
        tf.constant(
            [
                [4, 4, 2],
                [4, 2, 1],
                [3, 4, 2],
                [3, 2, 1],
                [4, 3, 4],
                [3, 1, 4],
                [4, 1, 1],
                [1, 4, 4],
                [2, 4, 1],
                [2, 1, 4],
            ]
        ),
    )
    npt.assert_allclose(
        partition_nd.front,
        tf.constant(
            [
                [0.0, 2.0, 1.0],
                [7.0, 6.0, 0.0],
                [9.0, 0.0, 1.0],
            ]
        ),
    )


def test_divide_conquer_non_dominated_raise_when_input_is_not_pareto_front():
    objectives = tf.constant(
        [
            [0.9575, 0.4218],
            [0.9649, 0.9157],
            [0.1576, 0.7922],
            [0.9706, 0.9595],
            [0.9572, 0.6557],
            [0.4854, 0.0357],
            [0.8003, 0.8491],
            [0.1419, 0.9340],
        ]
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        DividedAndConquerNonDominated(objectives)