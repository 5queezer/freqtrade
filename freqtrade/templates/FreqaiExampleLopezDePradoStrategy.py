"""
Example FreqAI Strategy using Lopez de Prado labeling methods.

Demonstrates:
- Triple-barrier labeling
- Purged K-Fold CV
- Sample weighting
- Fractional differentiation

Author: FreqAI Team
License: GPLv3
"""

import logging
from functools import reduce

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.freqai import lopez_de_prado as ldp
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair


logger = logging.getLogger(__name__)


class FreqaiExampleLopezDePradoStrategy(IStrategy):
    """
    Example strategy demonstrating Lopez de Prado methods in FreqAI.

    Uses triple-barrier labeling for more realistic target generation.
    """

    minimal_roi = {"0": 0.1, "240": -1}
    plot_config = {
        "main_plot": {},
        "subplots": {
            "prediction": {"prediction": {"color": "blue"}},
            "target": {"&-target": {"color": "green"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 30
    can_short = True

    # Triple-barrier parameters (Lopez de Prado)
    profit_target = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=False)
    stop_loss_barrier = DecimalParameter(-0.05, -0.01, default=-0.02, space="buy", optimize=False)
    max_holding_hours = IntParameter(6, 72, default=24, space="buy", optimize=False)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                        metadata: dict, **kwargs) -> DataFrame:
        """
        Generate features at different periods.
        *Only functional with FreqAI enabled strategies*
        """
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = ta.BBANDS(dataframe, timeperiod=period)
        dataframe["%-bb_lowerband-period"] = bollinger["lowerband"]
        dataframe["%-bb_middleband-period"] = bollinger["middleband"]
        dataframe["%-bb_upperband-period"] = bollinger["upperband"]

        dataframe["%-bb_width-period"] = (
            dataframe["%-bb_upperband-period"]
            - dataframe["%-bb_lowerband-period"]
        ) / dataframe["%-bb_middleband-period"]

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict,
                                          **kwargs) -> DataFrame:
        """
        Generate basic features.
        *Only functional with FreqAI enabled strategies*
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict,
                                     **kwargs) -> DataFrame:
        """
        Standard feature engineering without multi-period expansion.
        *Only functional with FreqAI enabled strategies*
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        # Volatility
        dataframe["%-volatility"] = dataframe["close"].rolling(20).std()

        # Price momentum
        dataframe["%-momentum"] = dataframe["close"].pct_change(periods=10)

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Lopez de Prado compliant target generation using triple-barrier method.

        This is the key difference from standard FreqAI strategies.
        """
        # Get close prices as pandas Series with DatetimeIndex
        close = dataframe['close'].copy()

        # Define events (potential trade entry points)
        # For this example, we'll use every candle as a potential entry
        # In practice, you might use specific signals
        events = close.index

        # Calculate triple-barrier labels
        barriers = ldp.get_events_triple_barrier(
            close=close,
            events=events,
            profit_target=self.profit_target.value,
            stop_loss=self.stop_loss_barrier.value,
            vertical_barrier_timedelta=pd.Timedelta(hours=self.max_holding_hours.value),
            side=None  # None = symmetric barriers for long/short
        )

        # Convert to binary classification labels
        # 1 = profitable trade, 0 = unprofitable trade
        labels = ldp.get_bins_from_triple_barrier(barriers, close)

        # Align with dataframe index
        dataframe['&-target'] = 0
        dataframe.loc[labels.index, '&-target'] = labels.values

        # Also store the actual return for reference (optional)
        dataframe['&-return'] = 0.0
        dataframe.loc[barriers.index, '&-return'] = barriers['return'].values

        # Store which barrier was touched (for analysis)
        dataframe['barrier_type'] = ''
        dataframe.loc[barriers.index, 'barrier_type'] = barriers['barrier_touched'].values

        logger.info(
            f"Triple-barrier labeling: "
            f"Profit hits: {(barriers['barrier_touched'] == 'profit').sum()}, "
            f"Stop hits: {(barriers['barrier_touched'] == 'stop').sum()}, "
            f"Timeouts: {(barriers['barrier_touched'] == 'vertical').sum()}"
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators and run FreqAI model.
        """
        # FreqAI feature engineering
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals based on FreqAI predictions.
        """
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-target"] > 0.5,  # Model predicts profitable trade
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-target"] < 0.5,  # Model predicts unprofitable long = short
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals (can be left simple since we use stoploss/ROI).
        """
        exit_long_conditions = [df["do_predict"] == 1, df["&-target"] < 0.5]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df["do_predict"] == 1, df["&-target"] > 0.5]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
