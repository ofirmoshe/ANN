import pytest
from numpy.testing import assert_array_equal
from unittest import TestCase
from ann import ANN
import numpy as np


class TestANN:

    def test__linear_forward(self):
        ann = ANN()
        x = np.array([[1, 2]]).transpose()
        w = np.array([[1, 3, 5], [2, 4, 6]]).transpose()
        b = np.array([[10, 11, 12]]).transpose()
        z, linear_cache = ann.linear_forward(A=x, W=w, b=b)
        exp_linear_cache = {'A': x, 'W': x, 'b': b}
        assert_array_equal(np.array([[15, 22, 29]]).transpose(), z, "linear_forward is wrong")
        all(np.array_equal(exp_linear_cache[key], linear_cache[key]) for key in exp_linear_cache)

    def test__soft_max(self):
        ann = ANN()
        z = np.array([[1, 1, 1]]).transpose()
        exp_res = np.array([[1 / 3] * 3]).transpose()
        a, activation_cache = ann.softmax(z)
        assert_array_equal(exp_res, a, "softmax is wrong")
        assert_array_equal(z, activation_cache, "activation cache of softmax is wrong")

    def test__relu(self):
        ann = ANN()
        z = np.array([[1.2, -2]]).transpose()
        a, activation_cache = ann.relu(z)
        assert_array_equal(a, np.array([[1.2, 0.]]).transpose())
        assert_array_equal(activation_cache, z)

    def test__linear_activation_forward(self):
        ann = ANN()
        x = np.array([[1, 2]]).transpose()
        w = np.array([[1, 3, 5], [2, 4, 6]]).transpose()
        b1 = np.array([[10, 11, 12]]).transpose()
        b2 = np.array([[-10, 11, 12]]).transpose()

        # test activation == softmax
        a1, cache = ann.linear_activation_forward(A_prev=x, W=w, B=b1, activation='softmax')
        assert np.sum(a1) == 1.0
        assert_array_equal(cache['activation_cache'], np.array([[15, 22, 29]]).transpose())
        exp_linear_cache_1 = {'A': x, 'W': x, 'b': b1}
        all(np.array_equal(exp_linear_cache_1[key], cache['linear_cache'][key]) for key in exp_linear_cache_1)

        # test activation == relu
        a2, cache = ann.linear_activation_forward(A_prev=x, W=w, B=b2, activation='relu')
        assert_array_equal(a2, np.array([[0, 22, 29]]).transpose())
        assert_array_equal(cache['activation_cache'], np.array([[-5, 22, 29]]).transpose())
        exp_linear_cache_2 = {'A': x, 'W': x, 'b': b2}
        all(np.array_equal(exp_linear_cache_2[key], cache['linear_cache'][key]) for key in exp_linear_cache_2)

        # test activation == other
        try:
            a1, cache = ann.linear_activation_forward(A_prev=x, W=w, B=b1, activation='other activation')
        except KeyError:
            assert True
        except Exception:
            assert False

    def test__L_model_forward(self):
        ann = ANN()
        x = np.array([[1, 2], [3,4]]).transpose()
        w1 = np.array([[1, 3, 5], [2, 4, 6]]).transpose()
        b1 = np.array([[10, 11, 12]]).transpose()
        w2 = np.array([[1, 3, 5], [2, 4, 6], [1, 2, 3]]).transpose()
        b2 = np.array([[10, 11, 12]]).transpose()
        a2, cache = ann.L_model_forward(X=x, parameters=dict({"W1":w1, "W2": w2, "b1": b1, "b2": b2}), use_batchnorm=False)
        exp_a2 = np.array([[4.64245566e-091, 6.51976599e-145],
                           [6.81355682e-046, 8.07450679e-073],
                           [1.00000000e+000, 1.00000000e+000]])
        #exp_linear_cache = {'A': x, 'W': x, 'b': b}
        assert_array_equal(np.round(exp_a2, decimals=2), np.round(a2, decimals=2), "L_model_forward is wrong")
        #all(np.array_equal(exp_linear_cache[key], linear_cache[key]) for key in exp_linear_cache)

    def test__compute_cost(self):
        AL = np.array([[0.15, 0.15],
                        [0.6, 0.6],
                        [0.25, 0.25]])
        Y = np.array([[0, 0],
                       [0, 0],
                       [1, 1]])
        assert(ANN.compute_cost(AL, Y) == 2)