# Important types

## Timing

```
alpaca.trading.enums.TimeInForce
    CLS = <TimeInForce.CLS: 'cls'>
    DAY = <TimeInForce.DAY: 'day'> 
    FOK = <TimeInForce.FOK: 'fok'>
    GTC = <TimeInForce.GTC: 'gtc'> // use with crypto
    IOC = <TimeInForce.IOC: 'ioc'> // also good for crypto
    OPG = <TimeInForce.OPG: 'opg'>
```

## Position Intent

```
alpaca.trading.enums.PositionIntent
    BUY_TO_CLOSE = <PositionIntent.BUY_TO_CLOSE: 'buy_to_close'>
    BUY_TO_OPEN = <PositionIntent.BUY_TO_OPEN: 'buy_to_open'>
    SELL_TO_CLOSE = <PositionIntent.SELL_TO_CLOSE: 'sell_to_close'>
    SELL_TO_OPEN = <PositionIntent.SELL_TO_OPEN: 'sell_to_open'>
```


## Orders

### Order Class

```
alpaca.trading.enums.OrderClass
    BRACKET = <OrderClass.BRACKET: 'bracket'>
    OCO = <OrderClass.OCO: 'oco'>
    OTO = <OrderClass.OTO: 'oto'>
    MLEG = <OrderClass.MLEG: 'mleg'>
    SIMPLE = <OrderClass.SIMPLE: 'simple'> // use with crypto
```

### Order Side

```
alpaca.trading.enums.OrderSide
    BUY = <OrderSide.BUY: 'buy'>
    SELL = <OrderSide.SELL: 'sell'>
```

### Order Status

```
alpaca.trading.enums.OrderStatus
    ACCEPTED = <OrderStatus.ACCEPTED: 'accepted'>
    ACCEPTED_FOR_BIDDING = <OrderStatus.ACCEPTED_FOR_BIDDING: 'accepted_for_bidding'>
    CALCULATED = <OrderStatus.CALCULATED: 'calculated'>
    CANCELED = <OrderStatus.CANCELED: 'canceled'>
    DONE_FOR_DAY = <OrderStatus.DONE_FOR_DAY: 'done_for_day'>
    EXPIRED = <OrderStatus.EXPIRED: 'expired'>
    FILLED = <OrderStatus.FILLED: 'filled'>
    HELD = <OrderStatus.HELD: 'held'>
    NEW = <OrderStatus.NEW: 'new'>
    PARTIALLY_FILLED = <OrderStatus.PARTIALLY_FILLED: 'partially_filled'>
    PENDING_CANCEL = <OrderStatus.PENDING_CANCEL: 'pending_cancel'>
    PENDING_NEW = <OrderStatus.PENDING_NEW: 'pending_new'>
    PENDING_REPLACE = <OrderStatus.PENDING_REPLACE: 'pending_replace'>
    PENDING_REVIEW = <OrderStatus.PENDING_REVIEW: 'pending_review'>
    REJECTED = <OrderStatus.REJECTED: 'rejected'>
    REPLACED = <OrderStatus.REPLACED: 'replaced'>
    STOPPED = <OrderStatus.STOPPED: 'stopped'>
    SUSPENDED = <OrderStatus.SUSPENDED: 'suspended'>
```

### Order Type

```
alpaca.trading.enums.OrderType
    MARKET = <OrderType.MARKET: 'market'>
    LIMIT = <OrderType.LIMIT: 'limit'>
    STOP = <OrderType.STOP: 'stop'>
    STOP_LIMIT = <OrderType.STOP_LIMIT: 'stop_limit'>
    TRAILING_STOP = <OrderType.TRAILING_STOP: 'trailing_stop'>
```

### Order Requests

Base is `alpaca.trading.requests`: https://alpaca.markets/sdks/python/api_reference/trading/requests.html#requests
All take the form of an `OrderRequest`:

    OrderRequest(
        *,
        symbol: str,
        qty: Optional[float], # Number of shares to trade
        notional: Optional[float], # Base currency value of shares to trade . DOES NOT WORK WHEN 'qty' IS PROVIDED
        side: OrderSide, # whether to buy or sell
        type: OrderType, # execution logic: market, limit, etc.
        time_in_force: Optional[TimeInForce], # expiration logic
        extended_hours: Optional[float], # whether it can be executed during regular market hours
        client_order_id: Optional[str], # string to identify which client submitted order
        order_class: Optional[OrderClass],
        legs: Optional[List[OptionLegRequest]],
        take_profit: Optional[ProfitLossRequest] # for order with multiple legs, order to exit a profitable trade
        stop_loss: Optional[StopLossRequest], # for orders with multiple legs, order to exit a losing trade
        position_intent: Optional[PositionIntent] # enum to indicate desired position strategy
    )

Don't use `OrderRequest` as it's the base class of more user-friendly derived classes. Using `*args` to denote all those parameters above, each of the derived classes are:

    MarketOrderRequest(*args)
    LimitOrderRequest(
        *args,
        limit_price: Optional[float] # 'limit_price' is worst fill price for a limit or stop limit order 
    ) 
    StopLimitOrderRequest(
        *args,
        stop_price: float, # see above
        limit_price: float # see above
    )
    StopOrderRequest(
        *args, 
        stop_price: float # 'stop_price' is price at which order is converted to a market order or a stop limit is converted to a limit order
    )
    TrailingStopOrderRequest(
        *args,
        trail_price: Optional[float], # percent price difference by which trailing will stop
        trail_percent: Optional[float] # percent price difference by which trailing will stop
    )

The remaining subclasses are used for retrieving the status of different orders and possibly modifying or replacing them:

    GetOrdersRequest(
        status: Optional[QueryOrderStatus],
        limit: Optiona[int],
        after: Optional[datetime],
        until: Optional[datetime],
        direction: Optional[Sort],
        nested: Optional[bool],
        side: Optional[OrderSide],
        symbols: Optional[List[str]]
    )
    GetOrderByIdRequest(
        nested: Optional[bool]
    )
    ReplaceOrderRequest()
    TakeProfitRequest()
    StopLossRequest()
    OptionalLegRequest()
    ClosePositionRequest()
    GetAssetRequest()
    GetPortfolioHistoryRequest()
    GetCalendarRequest()

There are also convenience functions for watchlists:

    CreateWatchlistRequest()
    UpdateWatchlistRequest()

Potpourri:

    GetCorporateAnnouncementRequest()
    CancelOrderResponse()


```
alpaca.trading.requests
    MARKET = <OrderType.MARKET: 'market'>
    LIMIT = <OrderType.LIMIT: 'limit'>
    STOP = <OrderType.STOP: 'stop'>
    STOP_LIMIT = <OrderType.STOP_LIMIT: 'stop_limit'>
    TRAILING_STOP = <OrderType.TRAILING_STOP: 'trailing_stop'>
```