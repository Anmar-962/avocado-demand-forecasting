"""Regression checks for the original project's forecasting pitfalls."""
import unittest
import numpy as np
import pandas as pd
from forecast import monday_start, feature_row, recursive_forecast, weekly_series


class ForecastTests(unittest.TestCase):
    def test_monday_boundary(self):
        dates = pd.Series(pd.to_datetime(['2023-08-14', '2023-08-20', '2023-08-21']))
        self.assertEqual(monday_start(dates).dt.strftime('%Y-%m-%d').tolist(), ['2023-08-14', '2023-08-14', '2023-08-21'])

    def test_features_use_only_prior_values(self):
        values = feature_row(pd.Timestamp('2023-08-21'), [10, 20, 30, 40])
        self.assertEqual(values[2:], [40, 30, 20, 10, 25])

    def test_recursive_prediction_feeds_next_step(self):
        class LastLagPlusOne:
            def predict(self, x):
                return np.array([x[0, 2] + 1])
        dates = pd.date_range('2023-08-21', periods=4, freq='W-MON')
        history = [1, 2, 3, 4]
        self.assertEqual(recursive_forecast(LastLagPlusOne(), history, dates), [5, 6, 7, 8])
        self.assertEqual(history, [1, 2, 3, 4])

    def test_missing_calendar_weeks_do_not_shift_lags(self):
        data = pd.DataFrame({'company':['A','A','B'], 'size':[32,32,48], 'week_start':pd.to_datetime(['2023-01-02','2023-01-16','2023-01-16']), 'quantity':[10,30,90]})
        weekly = weekly_series(data)
        a = weekly[weekly['company']=='A']
        b = weekly[weekly['company']=='B']
        self.assertEqual(a['quantity'].tolist(), [10, 0, 30])
        self.assertEqual(a['imputed_zero_week'].tolist(), [False, True, False])
        self.assertEqual(b['quantity'].tolist(), [90])

    def test_predictions_cannot_be_negative(self):
        class NegativeModel:
            def predict(self, x): return np.array([-3])
        self.assertEqual(recursive_forecast(NegativeModel(), [1,2,3,4], [pd.Timestamp('2023-01-02')]), [0])


if __name__ == '__main__':
    unittest.main()
