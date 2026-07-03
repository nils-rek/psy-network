"""Tests for the synthetic data generators."""

import numpy as np
import pandas as pd

from psynet.datasets import (
    make_bfi25,
    make_depression9,
    make_multigroup,
    make_multilevel_data,
    make_var_data,
)


class TestMakeBfi25:
    def test_shape_and_columns(self):
        df = make_bfi25(n=200)
        assert df.shape == (200, 25)
        assert list(df.columns)[:5] == ["O1", "O2", "O3", "O4", "O5"]

    def test_full_likert_range_used(self):
        """Standardized latent scores should populate the full 1-6 range."""
        df = make_bfi25(n=2000, seed=42)
        vals = df.values.ravel()
        assert vals.min() == 1
        assert vals.max() == 6
        # Tail categories should carry reasonable mass, not be vanishingly rare
        assert np.mean(vals == 1) > 0.02
        assert np.mean(vals == 6) > 0.02

    def test_reproducible(self):
        pd.testing.assert_frame_equal(make_bfi25(seed=7), make_bfi25(seed=7))

    def test_factor_structure_recoverable(self):
        """Items within a factor should correlate more than across factors."""
        df = make_bfi25(n=2000, seed=1)
        corr = df.corr()
        within = corr.loc["O1", "O2"]
        across = corr.loc["O1", "C1"]
        assert within > across + 0.1


class TestMakeDepression9:
    def test_shape_and_columns(self):
        df = make_depression9(n=150)
        assert df.shape == (150, 9)
        assert list(df.columns) == [f"dep{i}" for i in range(1, 10)]

    def test_all_categories_occur(self):
        """PHQ-9 style items must use the full 0-3 range, including 0."""
        df = make_depression9(n=2000, seed=123)
        vals = df.values.ravel()
        assert set(np.unique(vals)) == {0, 1, 2, 3}

    def test_roughly_uniform_marginals(self):
        """With uniform percentiles, each category gets ~25% of the mass."""
        df = make_depression9(n=5000, seed=0)
        vals = df.values.ravel()
        for cat in range(4):
            frac = np.mean(vals == cat)
            assert 0.15 < frac < 0.35

    def test_reproducible(self):
        pd.testing.assert_frame_equal(
            make_depression9(seed=9), make_depression9(seed=9),
        )


class TestMakeMultigroup:
    def test_shape(self):
        df = make_multigroup(n_per_group=100, n_groups=3, p=6)
        assert df.shape == (300, 7)
        assert set(df["group"]) == {"Group1", "Group2", "Group3"}

    def test_reproducible(self):
        pd.testing.assert_frame_equal(
            make_multigroup(seed=3), make_multigroup(seed=3),
        )

    def test_groups_differ(self):
        """Group-specific structure should make correlations differ."""
        df = make_multigroup(n_per_group=500, seed=42)
        groups = [g.drop(columns="group") for _, g in df.groupby("group")]
        c0 = groups[0].corr().values
        c1 = groups[1].corr().values
        assert np.abs(c0 - c1).max() > 0.05


class TestGeneratorsSmoke:
    def test_var_data_finite(self):
        df = make_var_data(n_timepoints=300, p=5, seed=1)
        assert np.all(np.isfinite(df.values))

    def test_multilevel_data_columns(self):
        df = make_multilevel_data(n_subjects=4, n_timepoints=20, p=3, seed=1)
        assert {"subject", "beep", "V1", "V2", "V3"} <= set(df.columns)
