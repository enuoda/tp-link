# Important types

```
alpaca.trading.enums.TimeInForce
    CLS = <TimeInForce.CLS: 'cls'>
    DAY = <TimeInForce.DAY: 'day'> 
    FOK = <TimeInForce.FOK: 'fok'>
    GTC = <TimeInForce.GTC: 'gtc'> // use with crypto
    IOC = <TimeInForce.IOC: 'ioc'> // also good for crypto
    OPG = <TimeInForce.OPG: 'opg'>
```

```
alpaca.trading.enums.PositionIntent
    BUY_TO_CLOSE = <PositionIntent.BUY_TO_CLOSE: 'buy_to_close'>
    BUY_TO_OPEN = <PositionIntent.BUY_TO_OPEN: 'buy_to_open'>
    SELL_TO_CLOSE = <PositionIntent.SELL_TO_CLOSE: 'sell_to_close'>
    SELL_TO_OPEN = <PositionIntent.SELL_TO_OPEN: 'sell_to_open'>
```


### Orders

```
alpaca.trading.enums.OrderClass
    BRACKET = <OrderClass.BRACKET: 'bracket'>
    OCO = <OrderClass.OCO: 'oco'>
    OTO = <OrderClass.OTO: 'oto'>
    MLEG = <OrderClass.MLEG: 'mleg'>
    SIMPLE = <OrderClass.SIMPLE: 'simple'> // use with crypto
```

```
alpaca.trading.enums.OrderSide
    BUY = <OrderSide.BUY: 'buy'>
    SELL = <OrderSide.SELL: 'sell'>
```


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

```
alpaca.trading.enums.OrderType
    MARKET = <OrderType.MARKET: 'market'>
    LIMIT = <OrderType.LIMIT: 'limit'>
    STOP = <OrderType.STOP: 'stop'>
    STOP_LIMIT = <OrderType.STOP_LIMIT: 'stop_limit'>
    TRAILING_STOP = <OrderType.TRAILING_STOP: 'trailing_stop'>
```