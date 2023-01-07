import unittest
import pandas as pd

from datetime import datetime
from backtrategy import backtrategy as bt


class DataSetTest(unittest.TestCase):

	def setUp(self):
		self.df = pd.DataFrame({
		    'time': ['20210101', '20210102', '20210103', '20210104', '20210105'],
		    'ask': [1.1, 1.2, 1.3, 1.4, 1.5],
		    'bid': [1.05, 1.15, 1.25, 1.35, 1.45]
		})

	def test_not_empty(self):
		data = bt.DataSet(self.df, bt.TickRepr('time', 'ask', 'bid', ''), bt.DataType.stock)
		self.assertIsNotNone(data.current_data)

	def test_first_element(self):
		data = bt.DataSet(self.df, bt.TickRepr('time', 'ask', 'bid', ''), bt.DataType.stock)
		current = data.current_data
		self.assertEqual(current['time'], datetime(2021, 1, 1))
		self.assertEqual(current['ask'], 1.1)
		self.assertEqual(current['bid'], 1.05)

	def test_last_element(self):
		data = bt.DataSet(self.df, bt.TickRepr('time', 'ask', 'bid', ''), bt.DataType.stock)
		current = data.current_data
		data.next_data()
		while data.current_data is not None:
			current = data.current_data
			data.next_data()

		self.assertEqual(current['time'], datetime(2021, 1, 5))
		self.assertEqual(current['ask'], 1.5)
		self.assertEqual(current['bid'], 1.45)

	def test_none_after_end(self):
		data = bt.DataSet(self.df, bt.TickRepr('time', 'ask', 'bid', ''), bt.DataType.stock)
		while data.current_data is not None:
			data.next_data()

		self.assertIsNone(data.current_data)


class Order(unittest.TestCase):

	def setUp(self):
		self.asset1_id = "eur/usd"
		self.asset1_type = bt.AssetType.cfd

	def test_order(self):
		order = bt.Order(
		    asset_id=self.asset1_id,
		    asset_type=self.asset1_type,
		    effect=bt.OrderEffect.open,
		    side=bt.OrderSide.long,
		    type=bt.OrderType.market,
		    quantity=1.1,
		    price=1000,  # TODO: Evaluate if give default value when market order
		    partial_fill=False)
		self.assertIsNotNone(order)
