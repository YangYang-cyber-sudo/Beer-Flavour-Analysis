"""Increment preprocessing arm used by the pre-registered head-to-head comparison.

Recovered from ``analysis_2026a/q1_v2/models.py`` in the original project.  It
extends :class:`modelling.models.FoldTransform` with a special-value indicator
and an optional source-group signed-log transform.  The estimator code is
unchanged; only the import path was adapted to this repository.
"""
import numpy as np

from .models import FoldTransform

# Sentinel used by the supplied table for "present but not numerically usable".
SPECIAL = -1.81781854


def special_mask(x):
    return np.isclose(x, SPECIAL, rtol=0, atol=5e-10)


class IncrementTransform(FoldTransform):
    def __init__(self, config, groups, verified_log, verified_identity):
        super().__init__(config, groups, verified_log)
        self.verified_identity = np.asarray(verified_identity, dtype=bool)
        if config.get('special', 'keep') not in ['keep', 'indicator']:
            raise ValueError('没有证据将特殊值改为缺失；当前协议只允许 keep/indicator')

    def _nonlinear(self, x):
        if self.config.get('shape') != 'source_group':
            return super()._nonlinear(x)
        out = x.copy()
        mask = self.verified_identity
        out[:, mask] = np.sign(out[:, mask]) * np.log1p(np.abs(out[:, mask]))
        return out

    def fit(self, x, y=None):
        # 特殊值指示的位置只由当前训练数据决定，不读取全量分布筛列。
        counts = special_mask(x).sum(axis=0)
        self.special_columns = np.flatnonzero((counts >= 2) & (counts < len(x)))
        super().fit(x, y)
        if self.config.get('scale') == 'by_source':
            z = np.clip(self._nonlinear(np.asarray(x, dtype=float)), self.low, self.high)
            z = np.where(np.isnan(z), self.median, z)
            robust = ~self.verified_log
            self.center[robust] = np.median(z[:, robust], axis=0)
            spread = (np.quantile(z[:, robust], .75, axis=0)
                      - np.quantile(z[:, robust], .25, axis=0))
            self.scale[robust] = np.where(spread > 1e-12, spread, 1.)
        return self

    def transform(self, x):
        out = super().transform(x)
        if self.config.get('special') == 'indicator':
            out = np.column_stack([out, special_mask(x)[:, self.special_columns].astype(float)])
        return out
