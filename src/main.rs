use std::cmp::Ordering;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum OrderType {
    Sell,
    Buy,
}

impl OrderType {
    fn opposite(&self) -> Self {
        match self {
            OrderType::Buy => OrderType::Sell,
            OrderType::Sell => OrderType::Buy,
        }
    }

    fn is_opposite(&self, other: &OrderType) -> bool {
        &self.opposite() == other
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Token {
    Usd,
    Btc,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TokenAmount {
    value: f64,
    token: Token,
}

impl PartialOrd for TokenAmount {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_same_token(other) {
            self.value.partial_cmp(&other.value)
        } else {
            None
        }
    }
}

impl std::fmt::Display for TokenAmount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2} {:?}", self.value, self.token)
    }
}

impl TokenAmount {
    fn is_same_token(&self, other: &TokenAmount) -> bool {
        self.token == other.token
    }

    fn gt_0(&self) -> bool {
        self.value > 0_f64
    }

    fn zero(token: Token) -> Self {
        TokenAmount { value: 0f64, token }
    }

    fn add(&self, other: &TokenAmount) -> Self {
        if self.token != other.token {
            panic!("Tokens do not match");
        }

        TokenAmount {
            value: self.value + other.value,
            token: self.token.clone(),
        }
    }

    fn sub(&mut self, other: &TokenAmount) -> Self {
        if self.token != other.token {
            panic!("Tokens do not match");
        }

        TokenAmount {
            value: self.value - other.value,
            token: self.token.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Market(Token, Token);

impl Market {
    fn opposite_token(&self, token: &Token) -> Token {
        if token == &self.0 {
            self.1
        } else {
            self.0
        }
    }

    fn contains_token(&self, token: &Token) -> bool {
        token == &self.0 || token == &self.1
    }

    fn allows_price(&self, price: &Price) -> bool {
        price.numerator != price.denominator
            && self.contains_token(&price.numerator)
            && self.contains_token(&price.denominator)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Price {
    numerator: Token,
    denominator: Token,
    value: f64,
}

impl PartialOrd for Price {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.has_same_tokens(other) {
            if self.numerator == other.numerator {
                self.value.partial_cmp(&other.value)
            } else {
                self.value.partial_cmp(&other.invert().value)
            }
        } else {
            None
        }
    }
}

impl std::fmt::Display for Price {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2} {:?}/{:?}",
            self.value, self.numerator, self.denominator
        )
    }
}

impl Price {
    fn invert(&self) -> Self {
        Price {
            numerator: self.denominator,
            denominator: self.numerator,
            value: 1_f64 / self.value,
        }
    }

    /// Multiply the price by the token amount, to get another token amount.
    ///
    /// If the token parameter is the same as the price denominator,
    /// then just multiply the values.
    /// If the token parameter is the same as the price numerator,
    /// then invert before multiplying.
    ///
    /// Panic if the token parameter does not match any of the price tokens.
    fn mult(&self, token_amount: &TokenAmount) -> TokenAmount {
        if token_amount.token == self.denominator {
            TokenAmount {
                value: self.value * token_amount.value,
                token: self.numerator,
            }
        } else if token_amount.token == self.numerator {
            TokenAmount {
                value: self.invert().value * token_amount.value,
                token: self.denominator,
            }
        } else {
            panic!("Invalid token");
        }
    }

    fn has_same_tokens(&self, other: &Price) -> bool {
        (self.denominator == other.numerator && self.numerator == other.denominator)
            || (self.denominator == other.denominator && self.numerator == other.numerator)
    }

    fn has_same_token(&self, token: &Token) -> bool {
        self.denominator == *token || self.numerator == *token
    }

    fn match_denominator_to(&self, token: &Token) -> Self {
        if self.denominator == *token {
            self.clone()
        } else if self.numerator == *token {
            self.invert()
        } else {
            panic!("Token does not match")
        }
    }
}

trait Order {
    fn id(&self) -> usize;
    fn market(&self) -> &Market;
    fn order_type(&self) -> &OrderType;
    fn token_amount(&self) -> &TokenAmount;
    fn token(&self) -> &Token;
    fn is_same_token(&self, token: &Token) -> bool;
    fn to_opposite(&self) -> Self;
}

#[derive(Debug, Clone, PartialEq)]
struct NonPricedOrder {
    id: usize,
    market: Market,
    order_type: OrderType,
    token_amount: TokenAmount,
}

#[derive(Debug, Clone)]
struct PricedOrder {
    id: usize,
    market: Market,
    order_type: OrderType,
    token_amount: TokenAmount,
    price: Price,
}

use std::sync::atomic::AtomicUsize;
static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl NonPricedOrder {
    fn new(market: Market, order_type: OrderType, token_amount: TokenAmount) -> Self {
        assert!(
            market.contains_token(&token_amount.token),
            "Token does not belong to the market"
        );

        use std::sync::atomic::Ordering;
        let id = ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        Self {
            id,
            market,
            order_type,
            token_amount,
        }
    }

    /// Convert from token A to token B.
    ///
    /// If this is a buy for token B, then it is also a sell for token A.
    /// And vice versa.
    fn invert(&self, price: &Price) -> Self {
        assert!(price.has_same_token(self.token()));

        Self {
            id: self.id,
            market: self.market,
            order_type: self.order_type().opposite(),
            token_amount: price.mult(self.token_amount()),
        }
    }

    /// Makes sure that the token for the order matches the parameter.
    fn convert_to(&self, token: &Token, price: &Price) -> Self {
        if self.is_same_token(token) {
            self.clone()
        } else {
            self.invert(price)
        }
    }
}

impl PricedOrder {
    fn new(market: Market, order_type: OrderType, token_amount: TokenAmount, price: Price) -> Self {
        assert!(
            market.allows_price(&price),
            "Price does not work with the market"
        );
        assert!(
            price.denominator == token_amount.token,
            "Price must have token as denominator"
        );
        assert!(
            market.contains_token(&token_amount.token),
            "Token does not belong to the market"
        );

        use std::sync::atomic::Ordering;
        let id = ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        Self {
            id,
            market,
            order_type,
            token_amount,
            price,
        }
    }

    /// Convert from token A to token B.
    ///
    /// If this is a buy for token B, then it is also a sell for token A.
    /// And vice versa.
    fn invert(&self) -> Self {
        Self {
            id: self.id,
            market: self.market,
            order_type: self.order_type().opposite(),
            token_amount: self.price.mult(self.token_amount()),
            price: self.price.invert(),
        }
    }

    /// Makes sure that the token for the order matches the parameter.
    fn convert_to(&self, token: &Token) -> Self {
        if self.is_same_token(token) {
            self.clone()
        } else {
            self.invert()
        }
    }

    /// Is one order a buy, and the other a sell (for the same token)?
    fn is_opposite<T: Order>(&self, other: &T) -> bool {
        self.convert_to(other.token())
            .order_type()
            .is_opposite(&other.order_type())
    }

    fn to_opposite_token_amount(&self) -> TokenAmount {
        self.price.mult(self.token_amount())
    }
}

impl Order for NonPricedOrder {
    fn id(&self) -> usize {
        self.id
    }

    fn market(&self) -> &Market {
        &self.market
    }

    fn order_type(&self) -> &OrderType {
        &self.order_type
    }

    fn token_amount(&self) -> &TokenAmount {
        &self.token_amount
    }

    fn token(&self) -> &Token {
        &self.token_amount().token
    }

    fn is_same_token(&self, token: &Token) -> bool {
        token == self.token()
    }

    fn to_opposite(&self) -> Self {
        Self::new(
            self.market.clone(),
            self.order_type.opposite(),
            self.token_amount.clone(),
        )
    }
}

impl Order for PricedOrder {
    fn id(&self) -> usize {
        self.id
    }

    fn market(&self) -> &Market {
        &self.market
    }

    fn order_type(&self) -> &OrderType {
        &self.order_type
    }

    fn token_amount(&self) -> &TokenAmount {
        &self.token_amount
    }

    fn token(&self) -> &Token {
        &self.token_amount().token
    }

    fn is_same_token(&self, token: &Token) -> bool {
        token == self.token()
    }

    fn to_opposite(&self) -> Self {
        Self::new(
            self.market.clone(),
            self.order_type.opposite(),
            self.token_amount.clone(),
            self.price.clone(),
        )
    }
}

#[derive(Debug, Clone)]
struct OrderBook {
    orders: Vec<PricedOrder>,
    market: Market,
}

impl OrderBook {
    fn new(orders: Vec<PricedOrder>) -> Self {
        assert!(orders.len() > 0, "Cannot have an empty order book");
        let market = orders.first().unwrap().market();

        assert!(
            orders
                .iter()
                .fold(true, |acc, order| acc && market == order.market()),
            "Order does not match market"
        );

        Self {
            market: market.clone(),
            orders,
        }
    }

    /// Average price of all orders, weighted by token amount.
    ///
    /// Assuming token_amounts are all for the same token, then this outputs:
    /// Sum(price * token_amount) / Sum(token_amount)
    fn weighted_average_price(&self, token: &Token) -> Price {
        let frac = self
            .orders
            .iter()
            .map(|order| order.convert_to(token)) // first convert all to same token
            .fold([0_f64, 0_f64], |acc, order| {
                // then calculate sum(opp_tokens)/sum(tokens)
                [
                    acc[0] + order.to_opposite_token_amount().value,
                    acc[1] + order.token_amount().value,
                ]
            });

        Price {
            value: frac[0] / frac[1],
            denominator: token.clone(),
            numerator: self.market.opposite_token(token),
        }
    }

    /// Calculate the average price that an order will be filled at.
    fn execution_price<T: Order + std::fmt::Debug>(&self, external_order: &T) -> Price {
        let token = external_order.token();

        // Opposite orders to external_order.
        let mut available_matches = self
            .orders
            .iter()
            // this clones the entire order book
            .map(|order| order.convert_to(token))
            .filter(|order| order.is_opposite(external_order))
            .collect::<Vec<PricedOrder>>();

        // Sort by price.
        available_matches.sort_by(|a, b| {
            let ord = a.price.partial_cmp(&b.price).unwrap();
            if external_order.order_type() == &OrderType::Buy {
                // We want lowest price for buy (ascending)..
                ord
            } else {
                // ..and highest price for sell (descending).
                ord.reverse()
            }
        });

        // Find the orders that exactly fill external_order.
        let exact_matches: Vec<PricedOrder> = available_matches
            .into_iter()
            .scan(0f64, |state, order| {
                if *state >= external_order.token_amount().value {
                    // Order has been filled already.
                    return None;
                }

                *state = *state + order.token_amount.value;

                if *state > external_order.token_amount().value {
                    // Order is filled, but only by a portion of the available match.
                    let mut order_mod = order.clone();
                    order_mod.token_amount.value = external_order.token_amount().value
                        - (*state - order_mod.token_amount.value);
                    Some(order_mod)
                } else {
                    // Order is not fully filled by the match (or is perfectly filled).
                    Some(order)
                }
            })
            .collect();

        OrderBook::new(exact_matches).weighted_average_price(token)
    }
}

/// Orders that have no price.
#[derive(Debug, Clone)]
struct SecondaryBook {
    orders: Vec<NonPricedOrder>,
    market: Market,
}

#[derive(Debug, Clone)]
enum MatchResult {
    PerfectlyFilled,
    SurplusBuys(Vec<NonPricedOrder>),
    SurplusSells(Vec<NonPricedOrder>),
}

impl PartialEq for MatchResult {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MatchResult::PerfectlyFilled, MatchResult::PerfectlyFilled) => true,
            (MatchResult::SurplusBuys(lhs), MatchResult::SurplusBuys(rhs)) => {
                if lhs.len() != rhs.len() {
                    false
                } else {
                    lhs.iter().zip(rhs.iter()).find(|(a, b)| *a != *b).is_none()
                }
            }
            (MatchResult::SurplusSells(lhs), MatchResult::SurplusSells(rhs)) => {
                if lhs.len() != rhs.len() {
                    false
                } else {
                    lhs.iter().zip(rhs.iter()).find(|(a, b)| *a != *b).is_none()
                }
            }
            (_, _) => false,
        }
    }
}

impl SecondaryBook {
    fn new(orders: Vec<NonPricedOrder>) -> Self {
        assert!(orders.len() > 0, "Cannot have an empty order set");
        let market = orders.first().unwrap().market();

        assert!(
            orders
                .iter()
                .fold(true, |acc, order| acc && market == order.market()),
            "Order does not match market"
        );

        Self {
            market: market.clone(),
            orders,
        }
    }

    fn match_orders(&self, price: &Price) -> MatchResult {
        assert!(self.market.allows_price(price));

        let token = price.denominator;

        let (mut buys, mut sells): (Vec<NonPricedOrder>, Vec<NonPricedOrder>) = self
            .orders
            .iter()
            .map(|order| order.convert_to(&token, price))
            .partition(|order| order.order_type() == &OrderType::Buy);

        // Sort descending by token amount.
        buys.sort_by(|a, b| b.token_amount.partial_cmp(a.token_amount()).unwrap());
        sells.sort_by(|a, b| b.token_amount.partial_cmp(a.token_amount()).unwrap());

        loop {
            let buy = buys.last_mut();
            let sell = sells.last_mut();

            // Match order to order until one of the vectors bottoms'out.
            match (buy, sell) {
                (Some(buy_order), Some(sell_order)) => {
                    if buy_order.token_amount == sell_order.token_amount {
                        buys.pop();
                        sells.pop();
                    } else if buy_order.token_amount > sell_order.token_amount {
                        buy_order.token_amount.value -= sell_order.token_amount.value;
                        sells.pop();
                    } else {
                        sell_order.token_amount.value -= buy_order.token_amount.value;
                        buys.pop();
                    }
                }
                (None, Some(_)) => return MatchResult::SurplusSells(sells),
                (Some(_), None) => return MatchResult::SurplusBuys(buys),
                (None, None) => return MatchResult::PerfectlyFilled,
            }
        }
    }

    /// Calculate the price that an order on the secondary book will execute at,
    /// given some order book.
    fn execution_price(&self, order_id: usize, order_book: &OrderBook) -> Price {
        assert_eq!(self.market, order_book.market, "Markets must match");

        let order = self
            .orders
            .iter()
            .find(|order| order.id == order_id)
            .expect("Order ID does not match any order in the secondary book")
            .clone();
        let token = order.token();
        let secondary_book_price = order_book.weighted_average_price(token);

        let calc_price = |orders: Vec<NonPricedOrder>, secondary_price: Price| {
            if orders.len() == 0 {
                panic!("Cannot have empty orders");
            }

            let outstanding_amount_for_order = orders
                .iter()
                .find(|order| order.id == order_id)
                .and_then(|order| Some(order.token_amount));

            if outstanding_amount_for_order.is_none() {
                secondary_price
            } else {
                let total_outstanding_amount = orders
                    .iter()
                    .fold(TokenAmount::zero(token.clone()), |acc, order| {
                        acc.add(order.token_amount())
                    });
                let settlement_order = NonPricedOrder::new(
                    self.market,
                    orders.first().unwrap().order_type,
                    total_outstanding_amount,
                );
                let order_book_price = order_book.execution_price(&settlement_order);

                let amount_matched_on_secondary = order
                    .token_amount
                    .clone()
                    .sub(&outstanding_amount_for_order.unwrap());

                let numerator = secondary_price
                    .mult(&amount_matched_on_secondary)
                    .add(&order_book_price.mult(&outstanding_amount_for_order.unwrap()));
                let denominator = order.token_amount;

                Price {
                    value: numerator.value / denominator.value,
                    numerator: self.market.opposite_token(token),
                    denominator: token.clone(),
                }
            }
        };

        match self.match_orders(&secondary_book_price) {
            MatchResult::PerfectlyFilled => secondary_book_price,
            MatchResult::SurplusBuys(buys) => calc_price(buys, secondary_book_price.clone()),
            MatchResult::SurplusSells(sells) => calc_price(sells, secondary_book_price.clone()),
        }
    }
}

fn analyse_price(order_book: &OrderBook, second_book: &SecondaryBook, numerator: &Token, denominator: &Token) {
    for i in 0..second_book.orders.len() {
        let order = second_book.orders.get(i).unwrap();
        let price_before = order_book.execution_price(order).match_denominator_to(denominator);
        let price_after = second_book
            .execution_price(order.id, &order_book)
            .match_denominator_to(denominator);

        let direction = if (order.order_type == OrderType::Buy && order.token() == denominator)
            || (order.order_type == OrderType::Sell && order.token() == numerator)
        {
            1_f64
        } else {
            -1_f64
        };
        let perc_diff =
            direction * 100_f64 * (price_before.value - price_after.value) / price_before.value;

        println!(
            "Order to {:?} {} has dex price {} and execution price {}, diff {:.2}%",
            order.order_type, order.token_amount, price_before, price_after, perc_diff
        );
    }
}

fn main() {
    let btc = Token::Btc;
    let usd = Token::Usd;
    let market = Market(btc, usd);

    let xbtc = |x: f64| {
        let btc = Token::Btc;

        TokenAmount {
            value: x,
            token: btc,
        }
    };

    let xusd = |x: f64| {
        let usd = Token::Usd;

        TokenAmount {
            value: x,
            token: usd,
        }
    };

    let usdperbtc = |x: f64| {
        let btc = Token::Btc;
        let usd = Token::Usd;

        Price {
            value: x,
            numerator: usd,
            denominator: btc,
        }
    };

    println!("\n===============\nOrder book BALANCED, 2nd book BALANCED\n");
    let order_book = OrderBook::new(vec![
        PricedOrder::new(market, OrderType::Sell, xbtc(10_f64), usdperbtc(64_f64)),
        PricedOrder::new(market, OrderType::Sell, xbtc(5_f64), usdperbtc(62_f64)),
        PricedOrder::new(market, OrderType::Sell, xbtc(3_f64), usdperbtc(60_f64)),
        PricedOrder::new(market, OrderType::Buy, xbtc(3_f64), usdperbtc(58_f64)),
        PricedOrder::new(market, OrderType::Buy, xbtc(5_f64), usdperbtc(56_f64)),
        PricedOrder::new(market, OrderType::Buy, xbtc(10_f64), usdperbtc(54_f64)),
    ]);
    let second_book = SecondaryBook::new(vec![
        NonPricedOrder::new(market, OrderType::Sell, xbtc(7_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(3_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(4_f64)),
    ]);
    analyse_price(&order_book, &second_book, &usd, &btc);

    println!("\n===============\nOrder book BALANCED, 2nd book ONLY BTC SELLS\n");
    let second_book = SecondaryBook::new(vec![
        NonPricedOrder::new(market, OrderType::Sell, xbtc(1_f64)),
        NonPricedOrder::new(market, OrderType::Sell, xbtc(5_f64)),
        NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
    ]);
    analyse_price(&order_book, &second_book, &usd, &btc);

    println!("\n===============\nOrder book BALANCED, 2nd book ONLY BTC BUYS\n");
    let second_book = SecondaryBook::new(vec![
        NonPricedOrder::new(market, OrderType::Buy, xbtc(1_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(5_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(10_f64)),
    ]);
    analyse_price(&order_book, &second_book, &usd, &btc);

    println!("\n===============\nOrder book BALANCED, 2nd book MORE BTC BUYS\n");
    let second_book = SecondaryBook::new(vec![
        NonPricedOrder::new(market, OrderType::Sell, xbtc(7_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(7_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(7_f64)),
    ]);
    analyse_price(&order_book, &second_book, &usd, &btc);

    // ===========================

    println!("\n===============\nOrder book MORE BTC SELLS, 2nd book BALANCED\n");
    let order_book = OrderBook::new(vec![
        PricedOrder::new(market, OrderType::Sell, xbtc(20_f64), usdperbtc(64_f64)),
        PricedOrder::new(market, OrderType::Sell, xbtc(10_f64), usdperbtc(62_f64)),
        PricedOrder::new(market, OrderType::Sell, xbtc(6_f64), usdperbtc(60_f64)),
        PricedOrder::new(market, OrderType::Buy, xbtc(3_f64), usdperbtc(58_f64)),
        PricedOrder::new(market, OrderType::Buy, xbtc(5_f64), usdperbtc(56_f64)),
        PricedOrder::new(market, OrderType::Buy, xbtc(10_f64), usdperbtc(54_f64)),
    ]);
    let second_book = SecondaryBook::new(vec![
        NonPricedOrder::new(market, OrderType::Sell, xbtc(7_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(3_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(4_f64)),
    ]);
    analyse_price(&order_book, &second_book, &usd, &btc);

    println!("\n===============\nOrder book MORE BTC SELLS, 2nd book ONLY BTC SELLS\n");
    let second_book = SecondaryBook::new(vec![
        NonPricedOrder::new(market, OrderType::Sell, xbtc(1_f64)),
        NonPricedOrder::new(market, OrderType::Sell, xbtc(5_f64)),
        NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
    ]);
    analyse_price(&order_book, &second_book, &usd, &btc);

    println!("\n===============\nOrder book MORE BTC SELLS, 2nd book ONLY BTC BUYS\n");
    let second_book = SecondaryBook::new(vec![
        NonPricedOrder::new(market, OrderType::Buy, xbtc(1_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(5_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(10_f64)),
    ]);
    analyse_price(&order_book, &second_book, &usd, &btc);

    println!("\n===============\nOrder book MORE BTC SELLS, 2nd book MORE BTC BUYS\n");
    let second_book = SecondaryBook::new(vec![
        NonPricedOrder::new(market, OrderType::Sell, xbtc(7_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(7_f64)),
        NonPricedOrder::new(market, OrderType::Buy, xbtc(7_f64)),
    ]);
    analyse_price(&order_book, &second_book, &usd, &btc);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xbtc(x: f64) -> TokenAmount {
        let btc = Token::Btc;

        TokenAmount {
            value: x,
            token: btc,
        }
    }

    fn xusd(x: f64) -> TokenAmount {
        let usd = Token::Usd;

        TokenAmount {
            value: x,
            token: usd,
        }
    }

    fn usdperbtc(x: f64) -> Price {
        let btc = Token::Btc;
        let usd = Token::Usd;

        Price {
            value: x,
            numerator: usd,
            denominator: btc,
        }
    }

    fn btcperusd(x: f64) -> Price {
        let btc = Token::Btc;
        let usd = Token::Usd;

        Price {
            value: x,
            numerator: btc,
            denominator: usd,
        }
    }

    fn order_book() -> OrderBook {
        let btc = Token::Btc;
        let usd = Token::Usd;
        let market = Market(btc, usd);

        OrderBook::new(vec![
            PricedOrder::new(market, OrderType::Sell, xbtc(10_f64), usdperbtc(20_f64)),
            PricedOrder::new(market, OrderType::Sell, xbtc(8_f64), usdperbtc(18_f64)),
            PricedOrder::new(market, OrderType::Sell, xbtc(6_f64), usdperbtc(16_f64)),
            PricedOrder::new(market, OrderType::Buy, xbtc(6_f64), usdperbtc(14_f64)),
            PricedOrder::new(market, OrderType::Buy, xbtc(8_f64), usdperbtc(12_f64)),
            PricedOrder::new(market, OrderType::Buy, xbtc(10_f64), usdperbtc(10_f64)),
        ])
    }

    #[test]
    fn order_token_equality() {
        let btc = Token::Btc;
        let usd = Token::Usd;
        let market = Market(btc, usd);

        let order = PricedOrder::new(market, OrderType::Buy, xbtc(10_f64), usdperbtc(3_f64));
        assert!(order.is_same_token(&Token::Btc));
    }

    #[test]
    fn price_inversion() {
        let p1 = usdperbtc(10_f64).invert();
        let p2 = btcperusd(0.1_f64);

        assert_eq!(p1, p2);
    }

    #[test]
    fn weighted_average_price() {
        let btc = Token::Btc;
        let usd = Token::Usd;
        let market = Market(btc, usd);

        let buy5 = PricedOrder::new(market, OrderType::Buy, xbtc(10_f64), usdperbtc(5_f64));
        let buy3 = PricedOrder::new(market, OrderType::Buy, xbtc(10_f64), usdperbtc(3_f64));
        let sell5 = PricedOrder::new(market, OrderType::Sell, xbtc(10_f64), usdperbtc(5_f64));
        let sell5_inverted = PricedOrder::new(
            market,
            OrderType::Sell,
            xusd(10_f64 * 5_f64),
            btcperusd(0.2_f64),
        );

        let book1 = OrderBook::new(vec![buy5, buy3.clone()]);
        let book2 = OrderBook::new(vec![sell5, buy3.clone()]);
        let book3 = OrderBook::new(vec![sell5_inverted, buy3]);
        let book4 = order_book();

        assert_eq!(book1.weighted_average_price(&btc), usdperbtc(4_f64));
        assert_eq!(book2.weighted_average_price(&btc), usdperbtc(4_f64));
        assert_eq!(book3.weighted_average_price(&btc), usdperbtc(4_f64));
        assert_eq!(book4.weighted_average_price(&btc), usdperbtc(15_f64));
    }

    #[test]
    fn execution_price() {
        let book = order_book();

        let priced = PricedOrder::new(
            book.market.clone(),
            OrderType::Sell,
            xbtc(6_f64),
            usdperbtc(16_f64),
        );
        let non_priced = NonPricedOrder::new(book.market.clone(), OrderType::Sell, xbtc(6_f64));

        assert_eq!(
            book.execution_price(&priced.to_opposite()),
            usdperbtc(16_f64)
        );
        assert_eq!(
            book.execution_price(&non_priced.to_opposite()),
            usdperbtc(16_f64)
        );

        let big_buy_btc = NonPricedOrder::new(book.market.clone(), OrderType::Buy, xbtc(20_f64));
        let p1 = (6 * 16 + 8 * 18 + 6 * 20) as f64 / 20_f64;
        assert_eq!(book.execution_price(&big_buy_btc), usdperbtc(p1));

        let small_buy_btc = NonPricedOrder::new(book.market.clone(), OrderType::Buy, xbtc(2_f64));
        assert_eq!(book.execution_price(&small_buy_btc), usdperbtc(16_f64));

        let big_sell_btc = NonPricedOrder::new(book.market.clone(), OrderType::Sell, xbtc(10_f64));
        let p2 = (6 * 14 + 4 * 12) as f64 / 10_f64;
        assert_eq!(book.execution_price(&big_sell_btc), usdperbtc(p2));

        let small_sell_btc = NonPricedOrder::new(book.market.clone(), OrderType::Sell, xbtc(2_f64));
        assert_eq!(book.execution_price(&small_sell_btc), usdperbtc(14_f64));
    }

    #[test]
    fn match_orders() {
        let btc = Token::Btc;
        let usd = Token::Usd;
        let market = Market(btc, usd);

        let price = usdperbtc(10f64);

        let perfectly_matched_btc = vec![
            NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
            NonPricedOrder::new(market, OrderType::Sell, xbtc(8_f64)),
            NonPricedOrder::new(market, OrderType::Buy, xbtc(6_f64)),
            NonPricedOrder::new(market, OrderType::Buy, xbtc(12_f64)),
        ];
        assert_eq!(
            SecondaryBook::new(perfectly_matched_btc).match_orders(&price),
            MatchResult::PerfectlyFilled
        );

        let perfectly_matched_usd_btc = vec![
            NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
            NonPricedOrder::new(market, OrderType::Sell, xusd(100_f64)),
        ];
        assert_eq!(
            SecondaryBook::new(perfectly_matched_usd_btc).match_orders(&price),
            MatchResult::PerfectlyFilled
        );

        let surplus_buys = vec![
            NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
            NonPricedOrder::new(market, OrderType::Buy, xbtc(6_f64)),
            NonPricedOrder::new(market, OrderType::Sell, xusd(100_f64)),
        ];
        let mut extra = surplus_buys.get(2).unwrap().clone();
        extra.token_amount = xbtc(6_f64);
        extra.order_type = OrderType::Buy;
        assert_eq!(
            SecondaryBook::new(surplus_buys).match_orders(&price),
            MatchResult::SurplusBuys(vec![extra])
        );

        let surplus_sells = vec![
            NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
            NonPricedOrder::new(market, OrderType::Buy, xbtc(6_f64)),
        ];
        let mut extra = surplus_sells.get(0).unwrap().clone();
        extra.token_amount = xbtc(4_f64);
        assert_eq!(
            SecondaryBook::new(surplus_sells).match_orders(&price),
            MatchResult::SurplusSells(vec![extra])
        );
    }

    #[test]
    fn secondary_book_execution_price() {
        let btc = Token::Btc;
        let usd = Token::Usd;

        let order_book = order_book();
        let market = order_book.market;

        let perfectly_matched_btc = vec![
            NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
            NonPricedOrder::new(market, OrderType::Sell, xbtc(8_f64)),
            NonPricedOrder::new(market, OrderType::Buy, xbtc(6_f64)),
            NonPricedOrder::new(market, OrderType::Buy, xbtc(12_f64)),
        ];
        let order = perfectly_matched_btc.first().unwrap().clone();
        let average_price = order_book.weighted_average_price(order.token());
        assert_eq!(
            SecondaryBook::new(perfectly_matched_btc).execution_price(order.id(), &order_book),
            average_price
        );

        let perfectly_matched_usd_btc = vec![
            NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
            NonPricedOrder::new(market, OrderType::Sell, xusd(100_f64)),
        ];
        let order = perfectly_matched_usd_btc.get(1).unwrap().clone();
        let average_price = order_book.weighted_average_price(order.token());
        assert_eq!(
            SecondaryBook::new(perfectly_matched_usd_btc).execution_price(order.id(), &order_book),
            average_price
        );

        let surplus_buys = vec![
            NonPricedOrder::new(market, OrderType::Sell, xbtc(10_f64)),
            NonPricedOrder::new(market, OrderType::Buy, xbtc(6_f64)),
            NonPricedOrder::new(market, OrderType::Sell, xusd(100_f64)),
        ];
        let order = surplus_buys.get(1).unwrap().clone();
        let average_price = order_book.weighted_average_price(order.token());
        assert_eq!(
            SecondaryBook::new(surplus_buys.clone()).execution_price(order.id(), &order_book),
            average_price
        );
        let order = surplus_buys.get(2).unwrap().clone();
        let average_price = order_book.weighted_average_price(order.token());
        let matched_btc = xbtc(4_f64);
        let surplus_btc = xbtc(6_f64);
        let order_book_price = usdperbtc(16_f64);
        let numerator = average_price
            .mult(&surplus_btc)
            .add(&order_book_price.mult(&matched_btc))
            .value;
        let denominator = xbtc(10_f64).value;
        let mut expected_price = Price {
            denominator: btc,
            numerator: usd,
            value: numerator / denominator,
        }
        .invert();
        expected_price.value = (expected_price.value * 1000_f64).round() / 1000_f64;
        assert_eq!(
            SecondaryBook::new(surplus_buys).execution_price(order.id(), &order_book),
            expected_price,
        );
    }
}
