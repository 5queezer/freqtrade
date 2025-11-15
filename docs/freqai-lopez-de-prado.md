# Lopez de Prado Methods

FreqAI implements machine learning methodologies from Marcos Lopez de Prado's "Advances in Financial Machine Learning" (2018) and "Machine Learning for Asset Managers" (2020).

## Features

### Purged K-Fold Cross-Validation

Prevents temporal leakage by:
- Respecting time order in fold splitting
- Purging overlapping samples between train/test
- Adding embargo periods to prevent look-ahead bias

```json
{
    "freqai": {
        "feature_parameters": {
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 5,
            "purged_cv_embargo_pct": 0.01,
            "label_horizon_candles": 10
        }
    }
}
```

### Sample Weighting

Multiple weighting strategies for better training:

**Time Decay** - Weight recent data more heavily:
```json
"ldp_time_decay": 1.0
```

**Sample Uniqueness** - Weight by information content (reduces overfitting from overlapping labels):
```json
"ldp_sample_uniqueness": true,
"label_horizon_candles": 10
```

**Return-Based** - Down-weight volatile periods:
```json
"ldp_return_weighting": true,
"ldp_return_weight_span": 60
```

### Fractional Differentiation

Makes features stationary while preserving memory. Better than traditional returns (d=1):

```json
"ldp_fractional_differentiation": true,
"ldp_frac_diff_order": 0.5,
"ldp_frac_diff_threshold": 0.01
```

Typical d values: 0.3-0.7. Lower = more memory preserved.

## Configuration Examples

### Minimal (Recommended Start)

```json
{
    "freqai": {
        "feature_parameters": {
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 5,
            "purged_cv_embargo_pct": 0.01,
            "ldp_time_decay": 1.0,
            "label_horizon_candles": 10
        }
    }
}
```

### Full Compliance

```json
{
    "freqai": {
        "feature_parameters": {
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 10,
            "purged_cv_embargo_pct": 0.02,
            "purged_cv_enable_purging": true,
            "label_horizon_candles": 20,
            "ldp_time_decay": 1.0,
            "ldp_sample_uniqueness": true,
            "ldp_fractional_differentiation": true,
            "ldp_frac_diff_order": 0.5
        }
    }
}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_purged_kfold_cv` | `false` | Enable Purged K-Fold CV |
| `purged_cv_n_splits` | `5` | Number of folds |
| `purged_cv_embargo_pct` | `0.01` | Embargo as % of dataset (1-5% typical) |
| `purged_cv_enable_purging` | `true` | Remove overlapping samples |
| `label_horizon_candles` | `10` | Forward-looking period of labels |
| `ldp_time_decay` | `0` | Time decay factor (0=off, 1.0=strong) |
| `ldp_sample_uniqueness` | `false` | Weight by uniqueness |
| `ldp_return_weighting` | `false` | Weight by volatility |
| `ldp_return_weight_span` | `60` | EMA span for volatility |
| `ldp_fractional_differentiation` | `false` | Enable fractional diff |
| `ldp_frac_diff_order` | `0.5` | Differentiation order d |
| `ldp_frac_diff_threshold` | `0.01` | Weight calculation threshold |
| `ldp_frac_diff_columns` | `null` | Columns to differentiate (null=all) |

## When to Use

**Always:**
- Purged K-Fold CV (if doing research/backtesting)
- Time decay weighting

**When Needed:**
- Fractional differentiation: Price features are non-stationary
- Sample uniqueness: Labels overlap significantly
- Return weighting: Multiple market regimes in dataset

## Performance Impact

| Feature | Time Impact | When to Use |
|---------|-------------|-------------|
| Purged K-Fold | +10-20% | Always for research |
| Fractional Diff | +5-10% | Price features |
| Sample Uniqueness | +15-25% | Overlapping labels |

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 4: Sample Weights
  - Chapter 5: Fractional Differentiation
  - Chapter 7: Cross-Validation in Finance
- Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
