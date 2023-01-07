import pandas as pd
import numpy as np
import talipp.indicators as tpi
import talipp.ohlcv as tpo

from enum import Enum
from decimal import *
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from queue import PriorityQueue
from typing import List
from typing import ClassVar
from ksuid import Ksuid


class DataType(Enum):
	ohlcv = 1
	tick = 2
	trade = 3


@dataclass
class DataType(Enum):
	stock = 1
	forex = 2
	crypto = 3


@dataclass
class DataStats:
	pass


@dataclass
class TickRepr:
	time: str
	ask: str
	bid: str
	volume: str


@dataclass
class Tick:
	time: datetime
	ask: Decimal
	bid: Decimal
	volume: int


@dataclass
class DataSet:
	data: pd.DataFrame
	repr: TickRepr
	type: DataType
	iterator: any = None
	current_data: dict = None

	def __post_init__(self) -> None:
		self.__start()

	def __start(self, timefmt: str = "") -> None:
		if self.repr.time not in self.data:
			raise ValueError("time column ", self.repr.time, " not found in data")

		if self.repr.ask not in self.data:
			raise ValueError("ask ", self.repr.ask, " not found in data")

		if self.repr.bid not in self.data:
			raise ValueError("bid", self.repr.bid, " not found in data")

		if self.repr.volume != "":
			if self.repr.volume not in self.data:
				raise ValueError("bid", self.repr.volume, " not found in data")

		if not np.issubdtype(self.data[self.repr.time].dtype, np.datetime64):
			if timefmt != "":
				self.data[self.repr.time] = pd.to_datetime(self.data[self.repr.time], format=timefmt)
			else:
				self.data[self.repr.time] = pd.to_datetime(self.data[self.repr.time])

		self.iterator = self.__generator()
		self.current_data = self.next_data()

	def reset_iterator(self) -> None:
		self.iterator = self.__generator()

	def get_current_data(self) -> dict:
		return self.current_data

	def get_current_field(self, field: str) -> any:
		return self.current_data[field]

	def __generator(self):
		for row in self.data.itertuples():
			yield row

	def next_data(self):
		try:
			self.current_data = next(self.iterator)._asdict()
			return self.current_data
		except StopIteration:
			self.current_data = None
			return None

	def current_time(self) -> np.datetime64:
		return self.current_data[self.repr.time]

	def set_data(self, data: pd.DataFrame, repr: Tick, timefmt: str = "") -> None:
		self.data = data
		self.repr = repr
		self.__start(timefmt)

	def is_active(self) -> bool:
		return self.iterator == None


class OrderType(Enum):
	market = 1
	limit = 2
	stop_limit = 3


class OrderEffect(Enum):
	open = 1
	close = 2
	partial_close = 3


class OrderSide(Enum):
	long = 1
	short = 2


class OrderState(Enum):
	open = 1
	executed = 2
	canceled = 3
	close = 4
	partial_filled = 5


class PositionState(Enum):
	open = 1
	closed = 2
	partial_closed = 3


class AssetType(Enum):
	shares = 1
	futures = 2
	cfd = 3
	crypto = 4


@dataclass
class Order:
	asset_id: str
	asset_type: AssetType
	effect: OrderEffect
	side: OrderSide
	type: OrderType
	quantity: Decimal
	price: Decimal
	partial_fill: bool
	create_time: np.datetime64 = np.datetime64('now')
	order_id: str = str(Ksuid())
	parent_order_id: str = ""
	state: OrderState = OrderState.open
	take_profit: Decimal = Decimal(0)
	stop_loss: Decimal = Decimal(0)
	executed_time: np.datetime64 = np.datetime64()
	cancel_time: np.datetime64 = np.datetime64()

	def __post_init__(self) -> None:
		if self.effect == OrderEffect.close or self.effect == OrderEffect.partial_close:
			if self.parent_order_id == "":
				raise ValueError("parent_order_id is required for close or partial_close order")

	def total_money(self) -> Decimal:
		return self.quantity * self.price

	def cancel_order(self) -> bool:
		if self.state == OrderState.open:
			self.state = OrderState.canceled
			self.cancel_time = np.datetime64()
			return True
		return False

	def execute_order(self,) -> bool:
		if self.state == OrderState.open:
			self.state = OrderState.executed
			self.executed_time = np.datetime64()
			return True
		return False

	def is_closed_order(self):
		return self.state == OrderState.close

	def is_open_order(self):
		return self.state == OrderState.open

	def is_executed_order(self):
		return self.state == OrderState.executed

	def is_canceled_order(self):
		return self.state == OrderState.canceled


@dataclass
class Position:
	open_order: Order
	quantity: Decimal(0)
	take_profit: Decimal = Decimal(0)
	stop_loss: Decimal = Decimal(0)
	state: PositionState = PositionState.open
	close_order: List[Order] = None

	def __post__init__(self):
		if self.open_order.effect != OrderEffect.open and self.open_order.state != OrderState.executed:
			raise ValueError("open_order must be an executed trade")

		self.quantity = self.open_order.quantity

	@property
	def position_id(self):
		return self.open_order.order_id

	@property
	def asset_id(self):
		return self.open_order.asset_id

	@property
	def asset_type(self):
		return self.open_order.asset_type

	@property
	def side(self):
		return self.open_order.side

	@property
	def quantity(self):
		return self.open_order.quantity

	@property
	def current_quantity(self):
		return self.quantity

	@property
	def open_price(self):
		return self.open_order.price

	@property
	def open_time(self):
		return self.open_order.executed_time

	def __validate_order(self, order) -> bool:
		if order.asset_id != self.asset_id or order.asset_type != self.asset_type or order.side == self.side or order.state != OrderState.executed:
			return False

		if order.effect == OrderEffect.partial_close and order.quantity > self.current_quantity:
			return False

		return True

	def partial_close(self, order: Order) -> bool:
		if self.state != PositionState.open:
			return False

		if not self.__validate_order(order):
			return False

		self.close_order.append(order)
		self.quantity -= order.quantity
		if self.quantity == 0:
			self.state = PositionState.closed
		else:
			self.state = PositionState.partial_closed
		return True

	def close(self, order: Order) -> bool:
		if self.state != PositionState.open:
			return False

		if not self.__validate_order(order):
			return False

		self.close_order.append(order)
		self.quantity = 0
		self.state = PositionState.closed
		return True
