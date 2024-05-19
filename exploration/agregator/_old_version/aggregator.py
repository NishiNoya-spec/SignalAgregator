from abc import ABC
from typing import List

import numpy as np


class Aggregator(ABC):

    @staticmethod
    def aggregate(array: np.array, **kwargs) -> np.array:
        ...


class DeltaAggregation(Aggregator):

    @staticmethod
    def aggregate(array: np.array, **kwargs):
        if len(array) == 0 or array[0] is None:
            return None
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return None
        else:
            return np.max(array) - np.min(array)


class MeanAggregation(Aggregator):

    @staticmethod
    def aggregate(array: np.array, **kwargs):
        if len(array) == 0 or array[0] is None:
            return None
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return None
        else:
            return np.mean(array)


class MedianAggregation(Aggregator):

    @staticmethod
    def aggregate(array: np.array, **kwargs):
        if len(array) == 0 or array[0] is None:
            return None
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return None
        else:
            return np.median(array)


class MaxAggregation(Aggregator):

    @staticmethod
    def aggregate(array: np.array, **kwargs):
        if len(array) == 0 or array[0] is None:
            return None
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return None
        else:
            return np.max(array)


class MinAggregation(Aggregator):

    @staticmethod
    def aggregate(array: np.array, **kwargs):
        if len(array) == 0 or array[0] is None:
            return None
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return None
        else:
            return np.min(array)


class StdAggregation(Aggregator):

    @staticmethod
    def aggregate(array: np.array, **kwargs):
        if len(array) == 0 or array[0] is None:
            return None
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return None
        else:
            return np.std(array)


class TgAggregation(Aggregator):

    @staticmethod
    def aggregate(array: np.array, points: np.array):
        if len(array) == 0 or array[0] is None:
            return None
        array = array[~np.isnan(array)]
        # Get the indexes of the NaN values in the original array
        nan_indexes = np.where(np.isnan(array))

        # Drop the NaN values from the array
        array_without_nan = array[~np.isnan(array)]
        points_without_nan = np.delete(points, nan_indexes)

        A = np.c_[points_without_nan, np.ones(len(points_without_nan))]
        p, _, _, _ = np.linalg.lstsq(A, array_without_nan, rcond=None)
        dots = A.dot(p)
        if len(dots) >= 2:
            y = dots[1] - dots[-1]
            x = points_without_nan[1] - points_without_nan[-1]
            return y / x if x != 0 else None
        else:
            return None


class PeriodAggregation(Aggregator):

    @staticmethod
    def get_paeks(array: np.array, points: np.array):
        signals = np.stack((array, points.T))

        signal_median = np.median(signals[0])
        signal_std = np.std(signals[0])
        cut_signals = signals[:, (signals[0] > signal_median + signal_std)]
        if cut_signals.shape[1] == 0:
            return []

        # находим пики сигнала
        distances_between_cut_signals = np.diff(cut_signals[1])
        distances_between_cut_signals = np.stack(
            (distances_between_cut_signals,
             np.array(range(len(distances_between_cut_signals))).T), )
        max_distances = distances_between_cut_signals[:, distances_between_cut_signals[
            0] > np.median(distances_between_cut_signals[0]
                           ) + np.std(distances_between_cut_signals[0])]

        peaks = []

        prev_index = 0
        try:
            for i in range(len(max_distances[1])):
                cur_index = int(max_distances[1][i])
                if cur_index != prev_index:
                    slice = cut_signals[0][prev_index:cur_index]
                    peaks.append(cut_signals[1][prev_index:cur_index][
                        slice == max(slice)][0])
                    prev_index = cur_index
        except:
            return []

        return peaks

    @staticmethod
    def get_periods(peaks: List, ):
        # вычисляем периоды волн
        if len(peaks) > 1:
            return [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
        return []

    @staticmethod
    def get_amplitudes(peaks: List, signal: np.array, points: np.array):
        # вычисляем амплитуды волн
        signals = np.stack((signal, points.T))
        signals = np.unique(signals, axis=1)
        amplitudes = []
        if len(peaks) > 1:
            for i in range(len(peaks) - 1):
                values = signals[0][(signals[1] >= peaks[i])
                                    & ([signals[1] <= peaks[i + 1]])[0]]
                amplitudes.append(max(values) - min(values))
        return amplitudes


class MinPeriodAggregation(PeriodAggregation):

    @classmethod
    def aggregate(cls, array: np.array, points: np.array):
        periods = cls.get_periods(cls.get_paeks(array, points))
        if len(periods) > 0:
            return min(periods)

        return 0


class MaxPeriodAggregation(PeriodAggregation):

    @classmethod
    def aggregate(cls, array: np.array, points: np.array):
        periods = cls.get_periods(cls.get_paeks(array, points))
        if len(periods) > 0:
            return max(periods)

        return 0


class MeanPeriodAggregation(PeriodAggregation):

    @classmethod
    def aggregate(cls, array: np.array, points: np.array):
        periods = cls.get_periods(cls.get_paeks(array, points))
        if len(periods) > 0:
            return sum(periods) / len(periods)
        return 0


class MinAmplitudeAggregation(PeriodAggregation):

    @classmethod
    def aggregate(cls, array: np.array, points: np.array):
        amplitudes = cls.get_amplitudes(
            cls.get_paeks(array, points),
            array,
            points,
        )
        if len(amplitudes) > 0:
            return min(amplitudes)
        return 0


class MaxAmplitudeAggregation(PeriodAggregation):

    @classmethod
    def aggregate(cls, array: np.array, points: np.array):
        amplitudes = cls.get_amplitudes(
            cls.get_paeks(array, points),
            array,
            points,
        )
        if len(amplitudes) > 0:
            return max(amplitudes)
        return 0


class MeanAmplitudeAggregation(PeriodAggregation):

    @classmethod
    def aggregate(cls, array: np.array, points: np.array):
        amplitudes = cls.get_amplitudes(cls.get_paeks(array, points), array,
                                        points)
        if len(amplitudes) > 0:
            return sum(amplitudes) / len(amplitudes)
        return 0


class FourierAggregation(PeriodAggregation):

    @staticmethod
    def aggregate(array: np.array):
        # Вычисление преобразования Фурье сигнала
        fft_signal = np.fft.fft(array)

        # Определение диапазона частот
        low_freq_range = (0, 15)  # низкие частоты
        high_freq_range = (15, 150)  # высокие частоты

        # Вычисление амплитудного спектра сигнала
        fft_amplitude = np.abs(fft_signal)

        # Вычисление частотного спектра сигнала
        N = len(array)
        fft_freq = np.fft.fftfreq(N, d=1 / N)

        # Анализ спектра на наличие интересующих частот
        low_freq_mask = np.logical_and(fft_freq >= low_freq_range[0],
                                       fft_freq <= low_freq_range[1])
        high_freq_mask = np.logical_and(fft_freq >= high_freq_range[0],
                                        fft_freq <= high_freq_range[1])
        try:
            if np.max(fft_amplitude[low_freq_mask]) > np.max(
                    fft_amplitude[high_freq_mask]):
                return 0  # низкочастотный сигнал
            else:
                return 1  # высокочастотный шум
        except ValueError:
            return 0  # низкочастотный сигнал


class AggregationFactory:

    def __init__(self):
        self.methods = (
            "median"
            "mean",
            "max",
            "min",
            "std",
            "tg",
            "min_period",
            "max_period",
            "mean_period",
            "min_amplitude",
            "max_amplitude",
            "mean_amplitude",
            "fourier",
            "delta"
        )
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self.methods):
            raise StopIteration
        value = self.methods[self._index]
        self._index += 1
        return value

    def __repr(self):
        return f'AggregationFactory(methods={self.methods})'

    @staticmethod
    def agg_by_method(array: np.array, method: str, **kwargs) -> np.array:
        if array.dtype not in ['float32', 'float64', 'int32', 'int64']:
            try:
                array = np.asarray(array, dtype=float)
            except:
                return None
        if method == "mean":
            return MeanAggregation.aggregate(array)
        elif method == "median":
            return MedianAggregation.aggregate(array)
        elif method == "max":
            return MaxAggregation.aggregate(array)
        elif method == "min":
            return MinAggregation.aggregate(array)
        elif method == "std":
            return StdAggregation.aggregate(array)
        elif method == "tg":
            return TgAggregation.aggregate(array, **kwargs)
        elif method == "min_period":
            return MinPeriodAggregation.aggregate(array, **kwargs)
        elif method == "max_period":
            return MaxPeriodAggregation.aggregate(array, **kwargs)
        elif method == "mean_period":
            return MeanPeriodAggregation.aggregate(array, **kwargs)
        elif method == "min_amplitude":
            return MinAmplitudeAggregation.aggregate(array, **kwargs)
        elif method == "max_amplitude":
            return MaxAmplitudeAggregation.aggregate(array, **kwargs)
        elif method == "mean_amplitude":
            return MeanAmplitudeAggregation.aggregate(array, **kwargs)
        elif method == "fourier":
            return FourierAggregation.aggregate(array)
        elif method == "delta":
            return DeltaAggregation.aggregate(array)
