# Lopez de Prado Machine Learning - FreqAI Integration Guide

## Overview

FreqAI now includes comprehensive support for **Marcos Lopez de Prado's** machine learning methodologies from his books:
- **"Advances in Financial Machine Learning" (2018)**
- **"Machine Learning for Asset Managers" (2020)**

These methods address critical issues in financial machine learning:
- **Temporal leakage** in cross-validation
- **Sample overlap** and path-dependency
- **Stationarity** while preserving information
- **Sample weighting** for better model training

## Key Features Implemented

### 1. Purged K-Fold Cross-Validation
✅ **Prevents look-ahead bias in time-series validation**

Traditional cross-validation randomly splits data, which can leak future information into training sets. Purged K-Fold:
- Respects temporal order
- Removes (purges) overlapping samples between train and test sets
- Adds embargo periods to prevent information leakage

### 2. Fractional Differentiation
✅ **Makes features stationary while preserving memory**

Traditional differencing (returns) removes all memory from the series. Fractional differentiation:
- Achieves stationarity with minimal d (differentiation order)
- Preserves more signal than integer differentiation
- Typical values: d = 0.3 to 0.7

### 3. Sample Uniqueness Weighting
✅ **Weights samples by information content**

Overlapping labels (common in financial ML) cause samples to share information. Sample uniqueness:
- Calculates how many other samples overlap with each sample
- Assigns higher weights to unique samples
- Reduces overfitting from redundant information

### 4. Time-Aware Sample Weights
✅ **Prioritizes recent, relevant data**

Combines multiple weighting strategies:
- Exponential time decay (more weight on recent data)
- Volatility-based weighting (down-weight extreme periods)
- Return-based weighting

### 5. Sequential Bootstrap
✅ **Resampling that respects temporal structure**

Standard bootstrap samples randomly. Sequential bootstrap:
- Samples with probability proportional to uniqueness
- Reduces overlap in bootstrapped samples
- Better preserves temporal dependencies

### 6. Combinatorial Purged Cross-Validation
✅ **Advanced validation for path-dependent strategies**

Creates multiple train/test combinations from time periods:
- Better variance estimation
- More robust performance metrics
- Ideal for strategy evaluation

---

## Configuration Guide

### Basic Configuration (Minimal Lopez de Prado)

Add to your FreqAI configuration:

```json
{
    "freqai": {
        "feature_parameters": {
            // Enable Purged K-Fold Cross-Validation
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 5,
            "purged_cv_embargo_pct": 0.01,

            // Enable Lopez de Prado time decay weighting
            "ldp_time_decay": 1.0,

            // Specify label horizon for purging calculations
            "label_horizon_candles": 10
        }
    }
}
```

### Advanced Configuration (Full Lopez de Prado Suite)

```json
{
    "freqai": {
        "feature_parameters": {
            // ===== CROSS-VALIDATION =====
            // Purged K-Fold CV with embargo
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 5,
            "purged_cv_embargo_pct": 0.02,  // 2% embargo period
            "purged_cv_enable_purging": true,
            "label_horizon_candles": 20,  // How far forward labels look

            // ===== SAMPLE WEIGHTING =====
            // Time decay (Lopez de Prado method)
            "ldp_time_decay": 1.0,  // 0 = uniform, 1.0 = exponential decay

            // Sample uniqueness weighting
            "ldp_sample_uniqueness": true,

            // Return-based weighting (down-weight volatile periods)
            "ldp_return_weighting": true,
            "ldp_return_weight_span": 60,  // EMA span for volatility

            // ===== FRACTIONAL DIFFERENTIATION =====
            // Make features stationary while preserving memory
            "ldp_fractional_differentiation": true,
            "ldp_frac_diff_order": 0.5,  // d value (0.3-0.7 typical)
            "ldp_frac_diff_threshold": 0.01,
            "ldp_frac_diff_columns": null,  // null = all numeric columns

            // ===== STANDARD FREQAI SETTINGS =====
            "indicator_periods_candles": [10, 20, 50],
            "include_timeframes": ["5m", "15m", "1h"],
            "include_shifted_candles": 2,
            "DI_threshold": 0,
            "use_SVM_to_remove_outliers": false,
            "principal_component_analysis": false
        },

        "data_split_parameters": {
            "test_size": 0.1,  // Ignored if use_purged_kfold_cv = true
            "shuffle": false   // Should always be false for time-series
        }
    }
}
```

---

## Configuration Parameters Reference

### Purged K-Fold Cross-Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_purged_kfold_cv` | bool | `false` | Enable Purged K-Fold CV instead of standard train_test_split |
| `purged_cv_n_splits` | int | `5` | Number of folds for cross-validation |
| `purged_cv_embargo_pct` | float | `0.01` | Embargo period as % of dataset (1-5% typical) |
| `purged_cv_enable_purging` | bool | `true` | Enable sample purging (remove overlapping samples) |
| `label_horizon_candles` | int | `10` | How many candles forward labels look (for purging) |

**Recommendations:**
- `purged_cv_n_splits`: 5-10 for most datasets
- `purged_cv_embargo_pct`: 0.01-0.05 (larger for higher-frequency strategies)
- `label_horizon_candles`: Should match your labeling function's forward-looking period

### Sample Weighting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ldp_time_decay` | float | `0` | Time decay factor (0=uniform, 1.0=strong decay) |
| `ldp_sample_uniqueness` | bool | `false` | Weight samples by uniqueness (inverse of overlap) |
| `ldp_return_weighting` | bool | `false` | Weight by return volatility |
| `ldp_return_weight_span` | int | `60` | EMA span for volatility calculation |

**Recommendations:**
- Use `ldp_time_decay` = 0.5-1.0 for markets with regime changes
- Enable `ldp_sample_uniqueness` when using overlapping labels
- Enable `ldp_return_weighting` to down-weight extreme market conditions

### Fractional Differentiation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ldp_fractional_differentiation` | bool | `false` | Enable fractional differentiation |
| `ldp_frac_diff_order` | float | `0.5` | Differentiation order d (0=none, 1=full diff) |
| `ldp_frac_diff_threshold` | float | `0.01` | Weight threshold for calculation |
| `ldp_frac_diff_columns` | list\|null | `null` | Specific columns to differentiate (null = all numeric) |

**Recommendations:**
- Start with `ldp_frac_diff_order` = 0.4-0.6
- Test stationarity with ADF test after applying
- Apply selectively to price-based features, not to volume/indicators
- Use `ldp_frac_diff_columns` to target specific features: `["close", "high", "low"]`

---

## Usage Examples

### Example 1: Conservative Setup (Prevent Overfitting)

Focus on preventing temporal leakage:

```json
{
    "freqai": {
        "feature_parameters": {
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 5,
            "purged_cv_embargo_pct": 0.02,
            "label_horizon_candles": 10,
            "ldp_time_decay": 0.5
        }
    }
}
```

### Example 2: Research-Grade Setup (Maximum Rigor)

Full Lopez de Prado compliance:

```json
{
    "freqai": {
        "feature_parameters": {
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 10,
            "purged_cv_embargo_pct": 0.05,
            "purged_cv_enable_purging": true,
            "label_horizon_candles": 20,

            "ldp_time_decay": 1.0,
            "ldp_sample_uniqueness": true,
            "ldp_return_weighting": true,
            "ldp_return_weight_span": 60,

            "ldp_fractional_differentiation": true,
            "ldp_frac_diff_order": 0.5,
            "ldp_frac_diff_threshold": 0.01
        }
    }
}
```

### Example 3: High-Frequency Trading

Short embargo, targeted differentiation:

```json
{
    "freqai": {
        "feature_parameters": {
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 8,
            "purged_cv_embargo_pct": 0.005,  // Smaller embargo for HFT
            "label_horizon_candles": 5,       // Short horizon

            "ldp_time_decay": 1.5,  // Strong recency bias
            "ldp_sample_uniqueness": true,

            "ldp_fractional_differentiation": true,
            "ldp_frac_diff_order": 0.4,  // Lighter differentiation
            "ldp_frac_diff_columns": ["close", "volume"]  // Targeted
        }
    }
}
```

---

## Performance Impact

### Computational Cost

| Feature | Training Time Impact | Memory Impact | Recommended Use |
|---------|---------------------|---------------|-----------------|
| Purged K-Fold CV | +10-20% | Low | Always for research |
| Fractional Differentiation | +5-10% | Low | Price features |
| Sample Uniqueness | +15-25% | Medium | Overlapping labels |
| Sequential Bootstrap | +20-30% | Medium | Advanced research |

### When to Use Each Feature

**Always Use:**
- Purged K-Fold CV (if doing research/backtesting)
- Time decay weighting

**Use When:**
- **Fractional Differentiation**: Price-based features are non-stationary
- **Sample Uniqueness**: Labels overlap significantly (e.g., 5-day returns calculated daily)
- **Return Weighting**: Dataset spans multiple market regimes (bull/bear/high volatility)

**Advanced/Optional:**
- Sequential Bootstrap: When bootstrapping for confidence intervals
- Combinatorial Purged CV: Final validation of deployed strategies

---

## Validation and Testing

### Verify Purged CV is Working

Check logs for:
```
INFO - Purged K-Fold split: train_size=8500, test_size=1500, embargo=1.0%
```

### Verify Fractional Differentiation

Check logs for:
```
INFO - Adding Fractional Differentiation to pipeline: d=0.5, threshold=0.01
```

### Compare Results

Run with and without Lopez de Prado features:

**Without (baseline):**
```json
{
    "feature_parameters": {
        "use_purged_kfold_cv": false,
        "ldp_fractional_differentiation": false
    }
}
```

**With (Lopez de Prado):**
```json
{
    "feature_parameters": {
        "use_purged_kfold_cv": true,
        "purged_cv_embargo_pct": 0.02,
        "ldp_fractional_differentiation": true,
        "ldp_frac_diff_order": 0.5
    }
}
```

**Expected differences:**
- Lower training accuracy (less overfitting)
- Better out-of-sample performance
- More stable live trading results

---

## Troubleshooting

### Issue: "ldp_time_decay requires DatetimeIndex"

**Solution:** Ensure your dataframe has a proper DatetimeIndex. This is standard in FreqAI but check if custom preprocessing removed it.

### Issue: Fractional differentiation creates NaN values

**Solution:**
- Reduce `ldp_frac_diff_order` (try 0.3-0.4)
- Increase `ldp_frac_diff_threshold` to 0.05
- Ensure you have enough historical data (at least 100+ samples)

### Issue: Training is very slow with sample uniqueness

**Solution:** Sample uniqueness is computationally expensive.
- Only enable for overlapping labels
- Consider using only `ldp_time_decay` for faster training

### Issue: Purged CV splits data unevenly

**Solution:** This is expected! Purging removes overlapping samples, which can create uneven splits. This is correct behavior and prevents leakage.

---

## References

### Books
1. **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
   - Chapter 4: Sample Weights
   - Chapter 5: Fractional Differentiation
   - Chapter 7: Cross-Validation in Finance
   - Chapter 12: Backtesting through Cross-Validation

2. **Lopez de Prado, M. (2020).** *Machine Learning for Asset Managers.* Cambridge University Press.
   - Additional insights on sample weighting and bootstrap methods

### Implementation Files
- `freqtrade/freqai/lopez_de_prado.py` - Core implementations
- `freqtrade/freqai/lopez_de_prado_transforms.py` - Pipeline transformers
- `freqtrade/freqai/data_kitchen.py` - Integration with data processing
- `freqtrade/freqai/freqai_interface.py` - Pipeline configuration

### Tests
- `tests/freqai/test_lopez_de_prado.py` - Unit tests for all methods

---

## Migration Guide

### Upgrading Existing Configs

**Old config (basic FreqAI):**
```json
{
    "feature_parameters": {
        "weight_factor": 0.5,
        "DI_threshold": 0
    },
    "data_split_parameters": {
        "test_size": 0.1,
        "shuffle": false
    }
}
```

**New config (Lopez de Prado compliant):**
```json
{
    "feature_parameters": {
        // Legacy weight_factor still works for backward compatibility
        "weight_factor": 0,  // Disable legacy, use ldp_time_decay instead

        // New Lopez de Prado features
        "use_purged_kfold_cv": true,
        "purged_cv_n_splits": 5,
        "purged_cv_embargo_pct": 0.01,
        "label_horizon_candles": 10,

        "ldp_time_decay": 1.0,
        "ldp_sample_uniqueness": true,
        "ldp_fractional_differentiation": true,
        "ldp_frac_diff_order": 0.5,

        "DI_threshold": 0
    },
    "data_split_parameters": {
        "test_size": 0.1,  // Ignored when use_purged_kfold_cv = true
        "shuffle": false
    }
}
```

**Backward Compatibility:**
- All existing configs continue to work
- Lopez de Prado features are opt-in
- Can enable incrementally (start with just purged CV, add others later)

---

## Best Practices

### 1. Start Simple
Begin with just Purged K-Fold CV, then add other features incrementally.

### 2. Match Embargo to Label Horizon
If your labels look 20 candles forward, your embargo should account for this overlap.

### 3. Validate Stationarity
Use ADF test to verify fractional differentiation achieves stationarity:
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(fractionally_differentiated_series)
print(f"ADF p-value: {result[1]}")  # Should be < 0.05
```

### 4. Monitor Training Metrics
Lopez de Prado methods typically:
- Reduce training accuracy (good - less overfitting)
- Improve validation accuracy
- Improve live trading performance

### 5. Document Your Choices
Record which Lopez de Prado features you use and why. This is crucial for research reproducibility.

---

## FAQ

**Q: Will this make my strategy more profitable?**
A: These methods reduce overfitting and improve validation. They help you avoid deploying bad strategies, but won't automatically make strategies profitable.

**Q: Should I use all features at once?**
A: No. Start with Purged K-Fold CV + time decay weighting. Add others if you have specific needs (overlapping labels → sample uniqueness, non-stationary features → fractional differentiation).

**Q: Does this work with RL models?**
A: Yes! The methods work with any model type (classification, regression, RL).

**Q: How much data do I need?**
A: Minimum 1000+ samples. For fractional differentiation, 500+ samples. More is better.

**Q: Can I use Combinatorial Purged CV?**
A: Yes, but it's not integrated into the main pipeline yet. Use `lopez_de_prado.CombinatorialPurgedKFold` directly in custom validation code.

---

## Support

For issues or questions:
1. Check logs for warnings/errors
2. Verify configuration parameters
3. Review test file for usage examples: `tests/freqai/test_lopez_de_prado.py`
4. Consult Lopez de Prado's books for theoretical background

**Happy training with scientifically rigorous ML! 🚀📊**
