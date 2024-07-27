//! # Pokers
//! A texas holdem poker library
//!
//! Currently supports
//!  - monte carlo range vs. range equity calculations
//!  - full enumeration for exact equities
//!  - fast hand evaluation
//!  - individual hand and combo results
//!
//! ## Equity Calculator
//!
//! ```
//! use std::sync::{atomic::AtomicBool, Arc};
//! use pokers::{HandRange, get_card_mask};
//! use pokers::approx_equity;
//! let ranges = HandRange::from_strings(["AK,22+".to_string(), "AA,KK,QQ@50".to_string()].to_vec());
//! let board_mask = get_card_mask("2h3d4c");
//! let dead_mask = get_card_mask("");
//! let cancel_token = Arc::new(AtomicBool::new(false));
//! let callback = |x: u8| {
//!     print!("\rProgress: {x}%");
//!     io::stdout().flush().unwrap();
//! };
//! let std_dev_target = 0.01;
//! let n_threads = 4;
//! let result = approx_equity(&ranges, board_mask, dead_mask, n_threads, std_dev_target, cancel_token, callback).unwrap();
//! let equities = result.equities;
//! ```
//!
//! ## Hand Evaluator
//!
//! ```
//! use pokers::*;
//! // cards are indexed 0->51 where index is 4 * rank + suit
//! let hand = Hand::default() + CARDS[0] + CARDS[1];
//! let score = hand.evaluate();
//! let board = get_card_mask("AhTd9d");
//! let hole_cards = Hand::from_hole_cards(44, 45);
//! let final_hand = board + hole_cards;
//! let final_score = final_hand.evaluate();
//! let rank = final_score / 4096; // or final_score >> 12
//! ```

pub mod constants;
pub mod hand_evaluator;
pub mod hand_range;
pub mod fastdivide;
pub mod equity_calculator;
pub mod string_lookup;

pub mod lookup;
pub mod flush_lookup;
pub mod hash_offsets;
pub mod map;

pub use hand_range::{HandRange, get_card_mask};
pub use equity_calculator::{exact_equity, approx_equity, SimulationResults, UnitResults, SimulatorError};
pub use hand_evaluator::*;
pub use string_lookup::*;
