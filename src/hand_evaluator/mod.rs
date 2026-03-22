mod hand;
mod cards;

pub use hand::{get_card_index, get_draw};
pub use cards::CARDS;

/// 64 bit representation of poker hand for use in evaluator
///
/// Bits 0-31: key to non flush lookup table
/// Bits 32-35: card counter
/// Bits 48-63: suit counter
/// Bits 64-128: Bit mask for all cards (suits in 16 bit groups)
#[derive(Debug, Copy, Clone)]
pub struct Hand {
    key: u64,
    mask: u64,
}
