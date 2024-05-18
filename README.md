
# Pokers

[![docs.rs](https://docs.rs/pokers/badge.svg)](https://docs.rs/pokers)
[![crates.io](https://img.shields.io/crates/v/pokers.svg)](https://crates.io/crates/pokers)

A poker library written in rust.

 - Multithreaded range vs range equity calculation
 - Fast hand evaluation
 - Efficient hand indexing
 - Individual hand and combo results


## Installation

Add this to your `Cargo.toml`:
```
[dependencies]
pokers = "0.3.1"
```
## Hand Evaluator

Evaluates the strength of any poker hand using up to 7 cards.

### Usage

```rust
use pokers::{Hand, CARDS};
// cards are indexed 0->51 where index is 4 * rank + suit
let hand = Hand::empty() + CARDS[0] + CARDS[1];
let score = hand.evaluate();
println!("score: {}", score);
```

## Equity Calculator

Calculates the range vs range equities for up to 6 different ranges specified by equilab-like range strings.
Supports monte-carlo simulations and exact equity calculations

### Usage

```rust
use std::sync::{atomic::AtomicBool, Arc};
use pokers::{HandRange, get_card_mask};
use pokers::approx_equity;
let ranges = HandRange::from_strings(["AK,22+".to_string(), "AA,KK,QQ@50".to_string()].to_vec());
let board_mask = get_card_mask("2h3d4c");
let dead_mask = get_card_mask("");
let cancel_token = Arc::new(AtomicBool::new(false));
let callback = |x: u8| {
    print!("\rProgress: {x}%");
    io::stdout().flush().unwrap();
};
let std_dev_target = 0.01;
let n_threads = 4;
let result = approx_equity(&ranges, board_mask, dead_mask, n_threads, std_dev_target, cancel_token, callback).unwrap();
let equities = result.equities;
println!("player 1 equity: {}", equities[0]);
```

## Credit

This library is a fork of **Kyle Murphy's** rust rewrite ([rust_poker](https://github.com/kmurf1999/rust_poker)) of **zekyll's** C++ equity calculator, [OMPEval](https://github.com/zekyll/OMPEval).
Differences with the original repo:
 - Fixes an issue that's been in the library for years without update. Impacts exact_equity()
 - It stores the results for the individual hands and then aggregates them into combos at the end of the simulation.
 - Only 1 dependency (rand).
 - Uses hardcoded arrays for lookups, so no need to read and write from files, use vectors, or lazy_static. This speeds up compilation significantly.
 - Adds a cancel token to the simulator to stop it at any time from another thread.
 - Uses a callback that takes the progress of the simulation.

## License

This project is MIT Licensed

Copyright (c) 2020 Kyle Murphy

Copyright (c) 2024 Eduardo Mata Ewy
