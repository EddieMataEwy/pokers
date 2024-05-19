use std::ops::Add;
use std::ops::AddAssign;

use crate::constants::*;
use crate::map::{CARD_MAP, MASK_MAP};
use crate::lookup::LOOKUP;
use crate::flush_lookup::LOOKUP_FLUSH;
use crate::hash_offsets::PERF_HASH_OFFSETS;
use crate::hand_evaluator::CARDS;

/// Return hole cards id
#[inline(always)]
pub fn get_card_index(card1: u8, card2: u8) -> usize {
    let mut card1 = card1;
    let mut card2 = card2;
    if card1 > card2 {
        std::mem::swap(&mut card1, &mut card2);
    }
    let index = card1 as usize * (101 - card1 as usize) / 2 + card2 as usize - 1;
    unsafe {*CARD_MAP.get_unchecked(index) as usize}
}

/// This function is very experimental.
/// Usage not recommended.
/// 1 - Straight Draw
/// 2 - Gutshot Draw
/// 4 - Double Gutshot Draw
/// 8 - Backdoor Straight Draw
/// 16 - Nut Straight 
/// 32 - Flush Draw
/// 64 - Backdoor Flush Draw
/// 128 - Nut (Backdoor) Flush (Draw)
pub fn get_draw(hole_cards: Hand, board: Hand, rank: u8) -> u8 {
    let mut draw: u8 = 0;
    let hand = hole_cards.clone() + board.clone();
    let rank_mask = hand.get_rank_mask();
    let get_backdoors = board.get_mask().count_ones() < 4;
    if rank < 4 {
        let mut oesd_mask: u16 = 0b1111;
        let mut bdsd_mask: u16 = 0b11111;
        let mut gsd_mask_1: u16 = 0b10111;
        let mut gsd_mask_2: u16 = 0b11011;
        let mut gsd_mask_3: u16 = 0b11101;
        let mut dgsd_mask_1: u16 = 0b1011101;
        let mut dgsd_mask_2: u16 = 0b11011011;
        for i in 0..10 {
            let masked_hand = rank_mask & oesd_mask;
            if masked_hand == oesd_mask {
                if i == 0 || i == 9 {
                    draw |= 2; // Gutshot Stratight Draw
                } else {
                    draw |= 1; // Open Ended Straight Draw
                }
            }
            if get_backdoors {
                let masked_hand = rank_mask & bdsd_mask;
                if masked_hand.count_ones() == 3 {
                    // Check for false positives
                    let check_1 = (rank_mask & (bdsd_mask << 1)).count_ones() > 3;
                    let check_2 = (rank_mask & (bdsd_mask >> 1)).count_ones() > 3;
                    if !check_1 && !check_2 {
                        draw |= 8; // Backdoor Stratight Draw
                    }
                }
            }
            let result_1 = (rank_mask & gsd_mask_1) == gsd_mask_1;
            let result_2 = (rank_mask & gsd_mask_2) == gsd_mask_2;
            let result_3 = (rank_mask & gsd_mask_3) == gsd_mask_3;
            if result_1 || result_2 || result_3 {
                draw |= 2; // Gutshot Straight Draw
            }
            let result_1 = (rank_mask & dgsd_mask_1) == dgsd_mask_1;
            let result_2 = (rank_mask & dgsd_mask_2) == dgsd_mask_2;
            if result_1 || result_2 {
                draw |= 4; // Gutshot Straight Draw
            }
            oesd_mask <<= 1;
            bdsd_mask <<= 1;
            gsd_mask_1 <<= 1;
            gsd_mask_2 <<= 1;
            gsd_mask_3 <<= 1;
            dgsd_mask_1 <<= 1;
            dgsd_mask_2 <<= 1;
        }
    }
    if rank == 4 {
        let mut mask: u16 = 0b1111100000000;
        let board_mask = board.get_rank_mask();
        for _ in 0..10 {
            let masked_board = board_mask & mask;
            if masked_board.count_ones() >= 3 {
                if (rank_mask & mask) == mask {
                    draw |= 16; // Nut Straight
                }
                // If board has 3 cards in the mask
                // but the hand is not a straight,
                // then it's not the nut straught.
                break;
            }
            mask >>= 1;
        }
    }
    if rank < 5 {
        for i in 0..4 {
            let suit_count = hand.suit_count(i);
            if suit_count  == 4 {
                draw |= 32; // Flush Draw
            } else if suit_count == 3 {
                if get_backdoors {
                    draw |= 64; // Backdoor Flush Draw
                }
            }
            if suit_count == 4 || (suit_count == 3 && get_backdoors) {
                let mask: u16 = 0b1111100000000;
                // Only the top 5 cards matter for the Nut Flush
                let board_mask = board.get_suit_mask(i) & mask;
                let hole_mask = hole_cards.get_suit_mask(i) & mask;
                // Which of the top 5 not in board?
                let xor_mask = (board_mask ^ mask) & mask;
                if xor_mask.leading_zeros() == hole_mask.leading_zeros() {
                    draw |= 128; // The player is holding the Nut Flush Draw 
                }
            }
        }
    }
    if rank == 5 {
        let mut suit = u8::MAX;
        for i in 0..4 {
            let suit_count = hand.suit_count(i);
            if suit_count >= 5 {
                suit = i;
                break; 
            }
        }
        let mask: u16 = 0b1111100000000;
        // Only the top 5 cards matter for the Nut Flush
        let board_mask = board.get_suit_mask(suit) & mask;
        let hole_mask = hole_cards.get_suit_mask(suit) & mask;
        // Which of the top 5 not in board?
        let xor_mask = (board_mask ^ mask) & mask;
        if xor_mask == 0 {
            draw |= 128; // Board is Nut Flush
        } else {
            if xor_mask.leading_zeros() == hole_mask.leading_zeros() {
                draw |= 128; // The player is holding the Nut Flush 
            }
        }
    }
    draw
}

/// 64 bit representation of poker hand for use in evaluator
///
/// Bits 0-31: key to non flush lookup table
/// Bits 32-35: card counter
/// Bits 48-63: suit counter
/// Bits 64-128: Bit mask for all cards (suits in 16 bit groups)
#[derive(Debug, Copy, Clone)]
pub struct Hand {
    pub key: u64,
    pub mask: u64,
}

impl Hand {
    /// Create hand from hole cards
    pub fn from_hole_cards(c1: u8, c2: u8) -> Hand {
        CARDS[usize::from(c1)] + CARDS[usize::from(c2)]
    }

    /// construct a Hand object from board mask
    pub fn from_bit_mask(mask: u64) -> Hand {
        let mut board = Hand::default();
        for c in 0..usize::from(CARD_COUNT) {
            if (mask & (1u64 << c)) != 0 {
                board += CARDS[c];
            }
        }
        board
    }

    /// Checks if hand contains another hand
    pub fn contains(&self, card: u8) -> bool {
        let mask = self.get_mask();
        let card_mask = 1u64 << ((3 - card % 4) * 16 + card / 4);
        mask & card_mask != 0
    }

    /// Return first 64 bits
    #[inline(always)]
    pub const fn get_key(self) -> u64 {
        self.key
    }
    /// Return last 64 bits
    #[inline(always)]
    pub const fn get_mask(self) -> u64 {
        self.mask
    }
    /// Returns 16 bit rank mask, ignoring suits
    pub fn get_suit_mask(&self, suit: u8) -> u16 {
        let hand_mask = self.get_mask();
        let mut rank_mask = 0u64;
        rank_mask |= (hand_mask >> 16 * (3 - suit)) & 0xFFFF;
        rank_mask as u16
    }
    /// Returns 16 bit rank mask, ignoring suits
    pub fn get_rank_mask(&self) -> u16 {
        let hand_mask = self.get_mask();
        let mut rank_mask = 0u64;
        for i in 0..4 {
            rank_mask |= (hand_mask >> 16 * (3 - i)) & 0xFFFF;
        }
        rank_mask as u16
    }
    /// get rank key of card for lookup table
    #[inline(always)]
    pub const fn get_rank_key(self) -> usize {
        // get last 32 bits
        let key = self.key as u32;
        // cast to usize
        key as usize
    }
    /// Return counter bits
    #[inline(always)]
    pub const fn get_counters(self) -> u32 {
        (self.key >> 32) as u32
    }
    /// Get flush key of card for lookup table
    ///
    /// Returns 0 if there is no flush
    #[inline(always)]
    pub fn get_flush_key(self) -> usize {
        // if hand has flush, return key
        // check to prevent throwing overflow error
        if self.has_flush() {
            // find which suit has flush
            let flush_check_bits = self.get_counters() & FLUSH_CHECK_MASK32;
            let shift = flush_check_bits.leading_zeros() << 2;
            // return mask for suit
            let key = (self.mask >> shift) as u16;
            usize::from(key)
        } else {
            0
        }
    }
    #[inline(always)]
    pub const fn has_flush(self) -> bool {
        (self.get_key() & FLUSH_CHECK_MASK64) != 0
    }
    // Return number of cards in hand
    #[inline(always)]
    pub const fn count(self) -> u32 {
        (self.get_counters() >> (CARD_COUNT_SHIFT - 32)) & 0xf
    }
    
    /// Get the number of cards for a suit
    #[inline(always)]
    pub const fn suit_count(self, suit: u8) -> i32 {
        let shift = 4 * suit + (SUITS_SHIFT - 32);
        (((self.get_counters() >> shift) & 0xf) as i32) - 3
    }
    
    /// Return hole cards id
     #[inline(always)]
     pub fn get_card_index(&self) -> usize {
         let mask = self.get_mask();
         let s = mask.trailing_zeros();
         let b = mask.leading_zeros();
         let b = 63 - b;
         let index = (b | (s<<6)) as usize;
         unsafe { *MASK_MAP.get_unchecked(index) as usize}
     }

    /// Returns hand strength in 16-bit integer.
    #[inline(always)]
    pub fn evaluate(&self) -> u16 {
        if self.has_flush() {
            unsafe { *LOOKUP_FLUSH.get_unchecked(self.get_flush_key()) }
        } else {
            let rank_key = self.get_rank_key();
            let offset = unsafe { *PERF_HASH_OFFSETS.get_unchecked(rank_key >> PERF_HASH_ROW_SHIFT) };
            let hash_key = (rank_key as u32).wrapping_add(offset) as usize;
            unsafe { *LOOKUP.get_unchecked(hash_key) }
        }
    }
    
    #[inline(always)]
    pub fn evaluate_without_flush(&self) -> u16 {
        let rank_key = self.get_rank_key();
        let offset = unsafe { *PERF_HASH_OFFSETS.get_unchecked(rank_key >> PERF_HASH_ROW_SHIFT) };
        let hash_key = (rank_key as u32).wrapping_add(offset) as usize;
        unsafe { *LOOKUP.get_unchecked(hash_key) }
    }
}

impl Default for Hand {
    // contruct the default hand
    // needed for evaluation
    // initializes suit counters
    //
    // # Example
    //
    // ```
    // use pokers::hand_evaluator::{Hand, CARDS};
    //
    // let hand = Hand::default() + CARDS[0] + CARDS[1];
    // let score = hand.evaluate();
    // ```
    fn default() -> Self {
        Hand {
            key: 0x3333u64 << SUITS_SHIFT,
            mask: 0,
        }
    }
}

impl Add for Hand {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            key: self.key + other.key,
            mask: self.mask | other.mask,
        }
    }
}

impl AddAssign for Hand {
    fn add_assign(&mut self, rhs: Hand) {
        self.key += rhs.key;
        self.mask |= rhs.mask;
    }
}

impl PartialEq for Hand {
    fn eq(&self, other: &Self) -> bool {
        (self.get_mask() == other.get_mask()) && (self.get_key() == other.get_key())
    }
}

impl Eq for Hand {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_constants() {
        // test a single card
        let rank: usize = 0; // 2
        let suit: usize = 0; // spade
        let h = CARDS[4 * rank + suit];
        assert_eq!(h.get_mask(), 1u64 << ((3 - suit) * 16 + rank));
        assert_eq!(h.count(), 1); // one card
        assert_eq!(h.has_flush(), false);
    }

    #[test]
    fn test_from_hole_cards() {
        // 2 of spades, 2 of hearts
        let h = Hand::from_hole_cards(0, 1);
        assert_eq!(h.count(), 2);
        assert_eq!(h.has_flush(), false);
    }

    #[test]
    fn test_rank_key() {
        // 2 of spades, 2 of hearts
        let h = Hand::from_hole_cards(0, 1);
        assert_eq!(h.get_rank_key() as u64, RANKS[0] + RANKS[0]);
    }

    #[test]
    fn test_flush_key() {
        let h_flush = Hand::default() + CARDS[0] + CARDS[4] + CARDS[8] + CARDS[12] + CARDS[16];
        assert_eq!(h_flush.get_flush_key(), 0b11111);

        let h_noflush = Hand::default() + CARDS[0] + CARDS[4] + CARDS[8] + CARDS[12];
        assert_eq!(h_noflush.get_flush_key(), 0);
    }

    #[test]
    fn test_has_flush() {
        let h_flush = Hand::default() + CARDS[0] + CARDS[8] + CARDS[12] + CARDS[16] + CARDS[20];
        assert_eq!(h_flush.has_flush(), true);
        let h_noflush = Hand::default() + CARDS[0] + CARDS[8] + CARDS[12] + CARDS[16] + CARDS[21];
        assert_eq!(h_noflush.has_flush(), false);
    }

    #[test]
    fn test_suit_count() {
        let h_4_spades = Hand::default() + CARDS[0] + CARDS[8] + CARDS[12] + CARDS[16] + CARDS[21];
        assert_eq!(h_4_spades.suit_count(0), 4);
        let h_3_hearts = Hand::default() + CARDS[1] + CARDS[9] + CARDS[13];
        assert_eq!(h_3_hearts.suit_count(1), 3);
    }
}
