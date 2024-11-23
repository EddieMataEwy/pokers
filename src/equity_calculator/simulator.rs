use std::ops::AddAssign;
use std::sync::mpsc::{channel, Sender};
use std::thread;
use std::sync::atomic::AtomicBool;

use crate::fastdivide::DividerU64;
use crate::get_draw;
use crate::string_lookup::{HAND_TO_INDEX, COMBO_TO_INDEX};
use std::cmp::Ordering;

use std::result::Result;
use std::sync::{Arc, Mutex, RwLock};

use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::{thread_rng, Rng, SeedableRng};

use super::CombinedRange;
use crate::constants::{CARD_COUNT, SUIT_COUNT, HAND_CATEGORY_OFFSET};
use crate::hand_evaluator::{get_card_index, Hand, CARDS};
use crate::hand_range::HandRange;

// use super::combined_range::CombinedRange;

const MIN_PLAYERS: usize = 2;
const MAX_PLAYERS: usize = 9;
const BOARD_CARDS: u32 = 5;
const NUM_HANDS: usize = 1326;
const NUM_COMBOS: usize = 169;

#[derive(Debug, Clone)]
pub enum SimulatorError {
    TooManyPlayers,
    TooFewPlayers,
    TooManyBoardCards,
    TooManyDeadCards,
    ConflictingRanges,
}

/// Calculates exact range vs range equities
///
/// Returns the equity for each player
///
/// # Arguments
///
/// * `hand_ranges` Array of hand ranges
/// * `board_mask` 64 bit mask of public cards
/// * `dead_mask` 64 bit mask of dead cards
/// * `n_threads` Number of threads to use in simulation
/// * `cancel_token` A shared boolean to stop simulation
/// * `callback` A function that takes estimated progress (0-100%)
///
/// # Example
/// ```
/// use std::sync::{atomic::AtomicBool, Arc};
/// use pokers::{HandRange, get_card_mask};
/// use pokers::approx_equity;
/// let ranges = HandRange::from_strings(["random".to_string(), "random".to_string()].to_vec());
/// let board_mask = get_card_mask("");
/// let dead_mask = get_card_mask("");
/// let cancel_token = Arc::new(AtomicBool::new(false));
/// let callback = |x: u8| {
///     print!("\rProgress: {x}%");
///     io::stdout().flush().unwrap();
/// };
/// let equities = exact_equity(&ranges, board_mask, dead_mask, 4, cancel_token, callback).unwrap();
/// ```
pub fn exact_equity<F: Fn(u8)>(
    hand_ranges: &[HandRange],
    board_mask: u64,
    dead_mask: u64,
    n_threads: u8,
    cancel_token: Arc<AtomicBool>,
    callback: F
) -> Result<SimulationResults, SimulatorError> {
    if hand_ranges.len() < MIN_PLAYERS {
        return Err(SimulatorError::TooFewPlayers);
    }
    if hand_ranges.len() > MAX_PLAYERS {
        return Err(SimulatorError::TooManyPlayers);
    }
    if board_mask.count_ones() > BOARD_CARDS {
        return Err(SimulatorError::TooManyBoardCards);
    }
    if 2 * hand_ranges.len() as u32 + dead_mask.count_ones() + BOARD_CARDS > CARD_COUNT as u32 {
        return Err(SimulatorError::TooManyDeadCards);
    }

    let mut hand_ranges = hand_ranges.to_owned();
    hand_ranges
        .iter_mut()
        .for_each(|h| h.remove_conflicting_combos(board_mask, dead_mask));
    let combined_ranges = CombinedRange::from_ranges(&hand_ranges);
    for cr in &combined_ranges {
        if cr.size() == 0 {
            return Err(SimulatorError::ConflictingRanges);
        }
    }

    let sim = Arc::new(Simulator::new(
        hand_ranges,
        combined_ranges,
        board_mask,
        dead_mask,
        true,
        0.0,
        cancel_token,
    ));

    let flag = sim.save_hand_weights();

    if board_mask.count_ones() > 2 {
        sim.set_ranks();
    }

    let (tx, rx) = channel();

    // spawn threads
    thread::scope(|scope| {
        for _ in 0..n_threads {
            let tx = tx.clone();
            let sim = Arc::clone(&sim);
            scope.spawn(move || {
                sim.enumerate_all(tx);
            });
        }
        drop(tx);
        for msg in rx.iter() {
            callback(msg);
        }
    });
    // get results and calculate equity
    sim.fix_hand_weights(flag);
    let results = Arc::try_unwrap(sim).unwrap().results;
    let mut results = results.into_inner().unwrap();
    results.get_equity();
    results.generate_final_data();
    if board_mask.count_ones() > 2 {
        results.generate_board_data();
    }
    Ok(results)
}

/// Runs a monte carlo simulation to calculate range vs range equity
///
/// Returns the equity for each player
///
/// # Arguments
///
/// * `hand_ranges` Array of hand ranges
/// * `board_mask` 64 bit mask of public cards
/// * `dead_mask` 64 bit mask of dead cards
/// * `n_threads` Number of threads to use in simulation
/// * `stdev_target` Target std deviation for simulation
/// * `cancel_token` A shared boolean to stop simulation
/// * `callback` A function that takes estimated progress (0-100%)
///
/// # Example
/// ```
/// use std::sync::{atomic::AtomicBool, Arc};
/// use pokers::{HandRange, get_card_mask};
/// use pokers::approx_equity;
/// let ranges = HandRange::from_strings(["random".to_string(), "random".to_string()].to_vec());
/// let board_mask = get_card_mask("");
/// let dead_mask = get_card_mask("");
/// let cancel_token = Arc::new(AtomicBool::new(false));
/// let callback = |x: u8| {
///     print!("\rProgress: {x}%");
///     io::stdout().flush().unwrap();
/// };
/// let equities = approx_equity(&ranges, board_mask, dead_mask, 4, 0.001, cancel_token, callback).unwrap();
/// ```
pub fn approx_equity<F: Fn(u8)>(
    hand_ranges: &[HandRange],
    board_mask: u64,
    dead_mask: u64,
    n_threads: u8,
    stdev_target: f64,
    cancel_token: Arc<AtomicBool>,
    callback: F
) -> Result<SimulationResults, SimulatorError> {
    if hand_ranges.len() < MIN_PLAYERS {
        return Err(SimulatorError::TooFewPlayers);
    }
    if hand_ranges.len() > MAX_PLAYERS {
        return Err(SimulatorError::TooManyPlayers);
    }
    if board_mask.count_ones() > BOARD_CARDS {
        return Err(SimulatorError::TooManyBoardCards);
    }
    if 2 * hand_ranges.len() as u32 + dead_mask.count_ones() + BOARD_CARDS > CARD_COUNT as u32 {
        return Err(SimulatorError::TooManyDeadCards);
    }
    
    let mut rng = thread_rng();
    let mut hand_ranges = hand_ranges.to_owned();
    hand_ranges
        .iter_mut()
        .for_each(|h| h.remove_conflicting_combos(board_mask, dead_mask));
    let mut combined_ranges = CombinedRange::from_ranges(&hand_ranges);
    for cr in &mut combined_ranges {
        if cr.size() == 0 {
            return Err(SimulatorError::ConflictingRanges);
        }
        cr.shuffle(&mut rng);
    }

    let sim = Arc::new(Simulator::new(
        hand_ranges,
        combined_ranges,
        board_mask,
        dead_mask,
        false,
        stdev_target,
        cancel_token,
    ));
    
    let flag = sim.save_hand_weights();

    if board_mask.count_ones() > 2 {
        sim.set_ranks();
    }

    let (tx, rx) = channel();
    
    // spawn threads
    thread::scope(|scope| {
        for _ in 0..n_threads {
            let tx = tx.clone();
            let sim = Arc::clone(&sim);
            let mut rng = SmallRng::from_rng(&mut rng).unwrap();
            scope.spawn(move || {
                sim.sim_random_walk_monte_carlo(&mut rng, tx);
            });
        }

        drop(tx);
        for msg in rx.iter() {
            callback(msg);
        }
    });
    // get results and calculate equity
    sim.fix_hand_weights(flag);
    let results = Arc::try_unwrap(sim).unwrap().results;
    let mut results = results.into_inner().unwrap();
    results.get_equity();
    results.generate_final_data();
    if board_mask.count_ones() > 2 {
        results.generate_board_data();
    }
    Ok(results)
}

/// Stores total results of the simulation
#[derive(Debug, Clone)]
pub struct SimulationResults {
    pub wins: Vec<u64>,
    pub ties: Vec<f64>,
    pub wins_by_mask: Vec<u64>,
    pub equities: Vec<f64>,
    pub eval_count: u64,
    pub batch_sum: f64,
    pub batch_sum2: f64,
    pub batch_count: f64,
    pub stdev: f64,
    /// All individual hands. Order is defined by `get_card_index()`
    /// [0-77] is pairs.
    /// [78-391] is suited hands.
    /// [392-1325] is offsuited hands.
    pub hand_results: Vec<UnitResults>,
    /// All preflob combos. Order derived by batching the hands.
    /// [0-12] is pairs.
    /// [13-91] is suited combos.
    /// [92-168] is offsuited combos.
    pub combo_results: Vec<UnitResults>,
    /// Result data for every rank.
    /// Ranks 0 -> 8 are High Card -> Straight Flush    
    pub board_results: Vec<UnitResults>,
    /// Result data for every draw and for nut straight and nut draw.
    /// Order is defined in `get_draw()` and `generate_board_data()`
    pub board_draws: Vec<UnitResults>,
}

impl SimulationResults {
    fn init(n_players: usize) -> SimulationResults {
        SimulationResults {
            wins: vec![0u64; n_players],
            ties: vec![0f64; n_players],
            wins_by_mask: vec![0u64; 1 << n_players],
            equities: vec![0.; n_players],
            eval_count: 0,
            batch_count: 0f64,
            batch_sum: 0f64,
            batch_sum2: 0f64,
            stdev: 0f64,
            hand_results: vec![UnitResults::default(); NUM_HANDS*n_players],
            combo_results: Vec::with_capacity(NUM_COMBOS*n_players),
            board_results: vec![UnitResults::default(); 9*n_players],
            board_draws: vec![UnitResults::default(); 10*n_players], 
        }
    }
    fn get_equity(&mut self) {
        let mut equity = vec![0f64; self.wins.len()];
        let mut equity_sum = 0f64;
        for i in 0..self.wins.len() {
            equity[i] += self.wins[i] as f64;
            equity[i] += self.ties[i];
            equity_sum += equity[i];
        }
        for e in &mut equity {
            *e /= equity_sum;
        }
        self.equities = equity;
    }
    pub fn get_hand_result(&self, player: usize,  hand: String) -> Result<UnitResults, &'static str> {
        let index = match HAND_TO_INDEX.get(hand.as_str()) {
            Some(i) => i,
            None => return Err("Invalid hand")
        };
        Ok(self.hand_results[NUM_HANDS*player+index].clone())
    }
    pub fn get_combo_result(&self, player: usize,  hand: String) -> Result<UnitResults, &'static str> {
        let index = match COMBO_TO_INDEX.get(hand.as_str()) {
            Some(i) => i,
            None => return Err("Invalid combo")
        };
        Ok(self.combo_results[NUM_COMBOS*player+index].clone())
    }
    fn generate_final_data(&mut self){
        let n_players = self.equities.len();
        for i in 0..n_players {
            let mut j: usize = 0;
            while j < NUM_HANDS {
                let mut counter = 0;
                let mut weight = 0;
                let mut wins: u128  = 0;
                let mut ties: u128  = 0;
                let mut corrected_ties = 0.;
                let mut total: u128  = 0;
                let batch_size = match j {
                    j if j < 78 => 6,
                    j if j < 390 => 4,
                    _ => 12
                };
                for k in 0..batch_size {
                    let hand_index = i*NUM_HANDS+j+k;
                    let hand = &mut self.hand_results[hand_index];
                    if hand.valid {
                        if hand.total > 0 {
                            hand.winrate = hand.wins as f64 / hand.total as f64;
                            hand.tierate = hand.ties as f64 / hand.total as f64;
                            counter += 1;
                            weight += hand.weight;
                            wins += hand.wins as u128;
                            ties += hand.ties as u128;
                            corrected_ties += hand.corrected_ties;
                            total += hand.total as u128;
                        } else {
                            hand.equity = 0.0;
                        }
                    }
                }
                if counter > 0 {
                    if total > u64::MAX as u128 {
                        let ratio = total / u64::MAX as u128;
                        let ratio = ratio + 1;
                        wins /= ratio;
                        ties /= ratio;
                        total /= ratio;
                        corrected_ties /= ratio as f64;
                    }
                    let combo = UnitResults {
                        wins: wins as u64,
                        ties: ties as u64,
                        total: total as u64,
                        corrected_ties,
                        equity: (wins as f64 + corrected_ties) / total as f64,
                        winrate: wins as f64 / total as f64,
                        tierate: ties as f64 / total as f64,
                        weight: weight/batch_size as u32,
                        rank: u8::MAX,
                        draw: 0,
                        valid: true
                    };
                    self.combo_results.push(combo);
                } else {
                    let combo = UnitResults::default();
                    self.combo_results.push(combo);
                }
                j += batch_size;
            }
        }
        self.combo_results.shrink_to_fit();
    }
    fn generate_board_data(&mut self) {
        let n_players = self.equities.len();
        let mut big_rank_data = vec![BigUnitResults::default();9*n_players];
        let mut big_draw_data = vec![BigUnitResults::default();10*n_players];
        for i in 0..n_players {
            for j in 0..NUM_HANDS {
                let hand = &self.hand_results[i*NUM_HANDS+j];
                if hand.valid {
                    let rank = hand.rank;
                    let rank_data = &mut big_rank_data[i*9+rank as usize];
                    rank_data.wins += hand.wins as u128;
                    rank_data.ties += hand.ties as u128;
                    rank_data.total += hand.total as u128;
                    rank_data.corrected_ties += hand.corrected_ties;
                    rank_data.weight += hand.weight;
                    let draw = hand.draw;
                    let mut mask = 1;
                    for k in 0..7 {
                        let masked_draw = draw & mask; 
                        if masked_draw != 0 {
                            let draw_data = &mut big_draw_data[i*10+k];
                            draw_data.wins += hand.wins as u128;
                            draw_data.ties += hand.ties as u128;
                            draw_data.total += hand.total as u128;
                            draw_data.corrected_ties += hand.corrected_ties;
                            draw_data.weight += hand.weight;
                        }
                        mask <<= 1;
                    }
                    let masked_draw = draw & mask;
                    if masked_draw != 0 {
                        let mut draw_rank = usize::MAX;
                        let check_1 = (draw & (mask >> 1)) != 0;
                        let check_2 = (draw & (mask >> 2)) != 0;
                        if check_1 {
                            draw_rank = 7; // Backdoor Nut Flush Draw
                        }
                        if check_2 {
                            draw_rank = 8; // Nut Flush Draw
                        }
                        if !check_1 && !check_2 {
                            draw_rank = 9; // Nut Flush
                        }
                        if draw_rank != usize::MAX {
                            let draw_data = &mut big_draw_data[i*10+draw_rank];
                            draw_data.wins += hand.wins as u128;
                            draw_data.ties += hand.ties as u128;
                            draw_data.total += hand.total as u128;
                            draw_data.corrected_ties += hand.corrected_ties;
                            draw_data.weight += hand.weight;
                        }
                    }
                }
            }
        }
        for (big_rank_data, rank_data) in big_rank_data.iter().zip(self.board_results.iter_mut()) {
            if big_rank_data.total > 0 {
                let ratio;
                if big_rank_data.total > u64::MAX as u128 {
                    ratio = big_rank_data.total / u64::MAX as u128 + 1;
                } else {
                    ratio = 1;
                }
                rank_data.wins = (big_rank_data.wins/ratio) as u64;
                rank_data.ties = (big_rank_data.ties/ratio) as u64;
                rank_data.total = (big_rank_data.total/ratio) as u64;
                rank_data.corrected_ties = big_rank_data.corrected_ties/ratio as f64;
                rank_data.weight = big_rank_data.weight;
                rank_data.equity = (rank_data.wins as f64 + rank_data.corrected_ties)/rank_data.total as f64;
                rank_data.winrate = rank_data.wins as f64/rank_data.total as f64;
                rank_data.tierate = rank_data.corrected_ties/rank_data.total as f64;
            }
        }
        for (big_draw_data, draw_data) in big_draw_data.iter_mut().zip(self.board_draws.iter_mut()) {
            if big_draw_data.total > 0 {
                let ratio;
                if big_draw_data.total > u64::MAX as u128 {
                    ratio = big_draw_data.total / u64::MAX as u128 + 1;
                } else {
                    ratio = 1;
                }
                draw_data.wins = (big_draw_data.wins/ratio) as u64;
                draw_data.ties = (big_draw_data.ties/ratio) as u64;
                draw_data.total = (big_draw_data.total/ratio) as u64;
                draw_data.corrected_ties = big_draw_data.corrected_ties/ratio as f64;
                draw_data.weight = big_draw_data.weight;
                draw_data.equity = (draw_data.wins as f64 + draw_data.corrected_ties)/draw_data.total as f64;
                draw_data.winrate = draw_data.wins as f64/draw_data.total as f64;
                draw_data.tierate = draw_data.corrected_ties/draw_data.total as f64;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct HandWithIndex {
    pub cards: (u8, u8, u8),
    pub player_idx: usize,
}

impl Default for HandWithIndex {
    fn default() -> Self {
        HandWithIndex {
            cards: (52, 52, 100),
            player_idx: 0,
        }
    }
}

impl Ord for HandWithIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        if (self.cards.0 >> 2) != (other.cards.0 >> 2) {
            return (self.cards.0 >> 2).cmp(&(other.cards.0 >> 2));
        }
        if (self.cards.1 >> 2) != (other.cards.1 >> 2) {
            return (self.cards.1 >> 2).cmp(&(other.cards.1 >> 2));
        }
        if (self.cards.0 & 3) != (other.cards.0 & 3) {
            return (self.cards.0 & 3).cmp(&(other.cards.0 & 3));
        }
        (self.cards.1 & 3).cmp(&(other.cards.1 & 3))
    }
}

impl PartialOrd for HandWithIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// This type can represent the results
/// of a single hand or a single combo.
#[derive(Debug, Clone)]
pub struct UnitResults {
    pub wins: u64,
    pub ties: u64,
    pub total: u64,
    pub corrected_ties: f64,
    pub equity: f64,
    pub winrate: f64,
    pub tierate: f64,
    pub weight: u32,
    pub rank: u8,
    pub draw: u8,
    pub valid: bool,
}

impl Default for UnitResults {
    fn default() -> Self {
        UnitResults { 
            wins: 0, 
            ties: 0, 
            total: 0, 
            corrected_ties: 0., 
            equity: 0., 
            winrate: 0., 
            tierate: 0., 
            weight: 0, 
            rank: u8::MAX, 
            draw: 0,
            valid: false
        }
    }
}

impl UnitResults {
    #[inline(always)]
    fn update_total(&mut self, weight: u64) {
        self.total += weight;
    }
    #[inline(always)]
    fn update_wins(&mut self, weight: u64) {
        self.wins += weight;
    }
    #[inline(always)]
    fn update_ties(&mut self, weight: u64, counter: usize) {
        self.ties += weight;
        self.corrected_ties += weight as f64 / counter as f64;
    }
}

impl AddAssign for UnitResults {
    fn add_assign(&mut self, other: Self) {
        self.wins += other.wins;
        self.ties += other.ties;
        self.total += other.total;
        self.corrected_ties += other.corrected_ties;
        self.equity = (self.wins as f64+self.corrected_ties)/self.total as f64;
    }
}

/// This type can aggregate more data
/// to avoid overflows.
#[derive(Debug, Clone)]
pub struct BigUnitResults {
    pub wins: u128,
    pub ties: u128,
    pub total: u128,
    pub corrected_ties: f64,
    pub weight: u32,
}

impl Default for BigUnitResults {
    fn default() -> Self {
        BigUnitResults { 
            wins: 0, 
            ties: 0, 
            total: 0, 
            corrected_ties: 0., 
            weight: 0, 
        }

    }
}

/// structure to store results of a single thread
#[derive(Debug, Copy, Clone)]
struct SimulationResultsBatch {
    wins_by_mask: [u64; 1 << MAX_PLAYERS],
    player_ids: [usize; MAX_PLAYERS],
    eval_count: u64,
}

impl SimulationResultsBatch {
    fn init(n_players: usize) -> SimulationResultsBatch {
        let mut player_ids = [0usize; MAX_PLAYERS];
        for i in 0..n_players {
            player_ids[i] = i;
        }
        SimulationResultsBatch {
            wins_by_mask: [0u64; 1 << MAX_PLAYERS],
            player_ids,
            eval_count: 0,
        }
    }
}

/// equity calculator main structure
#[derive(Debug)]
struct Simulator {
    hand_ranges: Vec<HandRange>,
    /// used to reduce rejection sampling
    combined_ranges: Vec<CombinedRange>,
    /// initial board as 64bit mask
    board_mask: u64,
    /// dead cards as 64bit mask
    dead_mask: u64,
    /// initial board used for evaluating
    fixed_board: Hand,
    /// number of players
    n_players: usize,
    /// has monte carlo sim stopped
    stopped: Arc<AtomicBool>,
    /// final results
    results: RwLock<SimulationResults>,
    /// target stdev from each batch for monte carlo
    stdev_target: f64,
    /// should calculate exact equity
    calc_exact: bool,
    /// preflop combo position for exact equity calculation
    enum_pos: Mutex<u64>,
}

impl Simulator {
    fn new(
        hand_ranges: Vec<HandRange>,
        combined_ranges: Vec<CombinedRange>,
        board_mask: u64,
        dead_mask: u64,
        calc_exact: bool,
        stdev_target: f64,
        cancel_token: Arc<AtomicBool>,
    ) -> Simulator {
        let fixed_board = Hand::from_bit_mask(board_mask);
        let n_players = hand_ranges.len();
        Simulator {
            hand_ranges,
            combined_ranges,
            board_mask,
            dead_mask,
            fixed_board,
            n_players,
            calc_exact,
            stopped: cancel_token,
            enum_pos: Mutex::new(0u64),
            results: RwLock::new(SimulationResults::init(n_players)),
            stdev_target,
        }
    }

    fn enumerate_all(&self, tx: Sender<u8>) {
        let mut enum_pos = 0u64;
        let mut enum_end = 0u64;
        let mut stats = SimulationResultsBatch::init(self.n_players);
        let mut hand_stats = vec![UnitResults::default(); self.n_players*NUM_HANDS];
        let fast_dividers: Vec<DividerU64> = self
            .combined_ranges
            .iter()
            .map(|c| DividerU64::divide_by(c.size() as u64))
            .collect();
        // let preflop_combos = self.get_preflop_combo_count();
        let postflop_combos = self.get_postflop_combo_count();

        // let randomize_order = postflop_combos > 10000 && preflop_combos <= 2 * MAX_LOOKUP_SIZE;
        loop {
            if enum_pos >= enum_end {
                let batch_size = std::cmp::max(2000000 / postflop_combos, 1);
                let (p, e) = self.reserve_batch(batch_size);
                enum_pos = p;
                enum_end = e;
                if enum_pos >= enum_end {
                    break;
                }
            }

            let mut rand_enum_pos = enum_pos;

            let mut ok = true;
            let mut used_cards_mask = self.board_mask | self.dead_mask;
            let mut player_hands = [HandWithIndex::default(); MAX_PLAYERS];
            for i in 0..self.combined_ranges.len() {
                let quotient = fast_dividers[i].divide(rand_enum_pos);
                let remainder = rand_enum_pos - quotient * self.combined_ranges[i].size() as u64;
                rand_enum_pos = quotient;
                let combo = &self.combined_ranges[i].combos()[remainder as usize];
                if (used_cards_mask & combo.mask) != 0 {
                    ok = false;
                    break;
                }
                used_cards_mask |= combo.mask;
                for j in 0..self.combined_ranges[i].player_count() {
                    let player_idx = self.combined_ranges[i].players()[j];
                    player_hands[player_idx].cards = combo.hole_cards[j];
                    player_hands[player_idx].player_idx = player_idx;
                }
            }

            if ok {
                let mut weight = 1u64;
                for hand in &player_hands[0..self.n_players] {
                    weight *= u64::from(hand.cards.2);
                }
                // stats.unique_preflop_combos += 1;
                self.enumerate_board(
                    &player_hands,
                    weight,
                    &self.fixed_board,
                    used_cards_mask,
                    &mut stats,
                    &mut hand_stats,
                );
            }

            if stats.eval_count >= 10000  {
                self.update_results(&tx, stats, hand_stats, false);
                stats = SimulationResultsBatch::init(self.n_players);
                hand_stats = vec![UnitResults::default(); self.n_players*NUM_HANDS];
                if self.stopped.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
            }
            enum_pos += 1;
        }

        self.update_results(&tx, stats, hand_stats, true);
    }

    fn enumerate_board(
        &self,
        player_hands: &[HandWithIndex],
        weight: u64,
        board: &Hand,
        used_cards_mask: u64,
        stats: &mut SimulationResultsBatch,
        hand_stats: &mut Vec<UnitResults>,
    ) {
        let mut hands = [Hand::default(); MAX_PLAYERS];
        let mut lookup = [0; MAX_PLAYERS];
        for i in 0..self.n_players {
            let card1 = player_hands[i].cards.0;
            let card2 = player_hands[i].cards.1;
            hands[i] = Hand::from_hole_cards(card1, card2);
            lookup[i] = NUM_HANDS*stats.player_ids[i]+get_card_index(card1, card2);
        }

        let cards_remaining = (BOARD_CARDS - board.count()) as u8;
        if cards_remaining == 0 {
            self.evaluate_hands(&hands, weight, board, &lookup, stats, hand_stats, true);
            return;
        }

        let mut deck = [0u8; 52];
        let mut n_deck = 0;
        for i in (0..CARD_COUNT).rev() {
            if (used_cards_mask & (1u64 << i)) == 0 {
                deck[n_deck] = i;
                n_deck += 1;
            }
        }

        let mut suit_counts = [0u8; 4];
        for i in 0..self.n_players {
            if (player_hands[i].cards.0 & 3) == (player_hands[i].cards.1 & 3) {
                suit_counts[usize::from(player_hands[i].cards.0 & 3)] =
                    std::cmp::max(2, suit_counts[usize::from(player_hands[i].cards.0 & 3)]);
            } else {
                suit_counts[usize::from(player_hands[i].cards.0 & 3)] =
                    std::cmp::max(1, suit_counts[usize::from(player_hands[i].cards.0 & 3)]);
                suit_counts[usize::from(player_hands[i].cards.1 & 3)] =
                    std::cmp::max(1, suit_counts[usize::from(player_hands[i].cards.1 & 3)]);
            }
        }
        for i in 0..SUIT_COUNT {
            suit_counts[usize::from(i)] += board.suit_count(i) as u8;
        }

        self.enumerate_board_rec(
            &hands,
            stats,
            hand_stats,
            &lookup,
            &board,
            &mut deck,
            n_deck,
            &mut suit_counts,
            cards_remaining,
            0,
            weight,
        );
    }

    fn enumerate_board_rec(
        &self,
        hands: &[Hand],
        stats: &mut SimulationResultsBatch,
        hand_stats: &mut Vec<UnitResults>,
        lookup: &[usize;MAX_PLAYERS],
        board: &Hand,
        deck: &mut [u8],
        n_deck: usize,
        suit_counts: &mut [u8],
        cards_remaining: u8,
        start: usize,
        weight: u64,
    ) {
        if cards_remaining == 1 {
            if (suit_counts[0] < 4)
                && (suit_counts[1] < 4)
                && (suit_counts[2] < 4)
                && (suit_counts[3] < 4)
            {
                let mut i = start;
                while i < n_deck {
                    let mut multiplier = 1;
                    let new_board = *board + CARDS[usize::from(deck[i])];
                    let rank = deck[i] >> 2;
                    i += 1;
                    while i < n_deck && deck[i] >> 2 == rank {
                        multiplier += 1;
                        i += 1;
                    }
                    self.evaluate_hands(hands, weight * multiplier, &new_board, lookup, stats, hand_stats, false);
                }
            } else {
                let mut last_rank = u8::MAX;
                for i in start..n_deck {
                    let mut multiplier = 1;
                    if suit_counts[usize::from(deck[i] & 3)] < 4 {
                        let rank = deck[i] >> 2;
                        if rank == last_rank {
                            continue;
                        }
                        for j in i + 1..n_deck {
                            if deck[j] >> 2 != rank {
                                break;
                            }
                            if suit_counts[usize::from(deck[j] & 3)] < 4 {
                                multiplier += 1;
                            }
                        }
                        last_rank = rank;
                    }
                    let new_board = *board + CARDS[usize::from(deck[i])];
                    self.evaluate_hands(hands, weight * multiplier, &new_board, lookup, stats, hand_stats, true);
                }
            }
            return;
        }
        let mut i = start;
        while i < n_deck {
            let mut new_board = *board;
            let suit = deck[i] & 3;
            if (suit_counts[usize::from(suit)] + cards_remaining) < 5 {
                let mut irrelevant_count = 1;
                let rank = deck[i] >> 2;
                for j in i + 1..n_deck {
                    if deck[j] >> 2 != rank {
                        break;
                    }
                    let suit2 = deck[j] & 3;
                    if (suit_counts[usize::from(suit2)] + cards_remaining) < 5 {
                        if j != i + irrelevant_count {
                            deck.swap(j, i + irrelevant_count);
                        }
                        irrelevant_count += 1;
                    }
                }

                for repeats in 1..std::cmp::min(irrelevant_count, usize::from(cards_remaining)) + 1
                {
                    const BINOM_COEFF: [[u64; 5]; 5] = [
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [1, 2, 1, 0, 0],
                        [1, 3, 3, 1, 0],
                        [1, 4, 6, 4, 1],
                    ];
                    let new_weight = BINOM_COEFF[irrelevant_count][repeats] * weight;
                    new_board += CARDS[usize::from(deck[i + repeats - 1])];
                    if repeats == usize::from(cards_remaining) {
                        self.evaluate_hands(&hands, new_weight, &new_board, lookup, stats, hand_stats, true);
                    } else {
                        self.enumerate_board_rec(
                            hands,
                            stats,
                            hand_stats,
                            lookup,
                            &new_board,
                            deck,
                            n_deck,
                            suit_counts,
                            cards_remaining - repeats as u8,
                            i + irrelevant_count,
                            new_weight,
                        );
                    }
                }

                i += irrelevant_count - 1;
            } else {
                // new_board.mask += u64::from(deck[i]);
                new_board += CARDS[usize::from(deck[i])];
                suit_counts[usize::from(suit)] += 1;
                self.enumerate_board_rec(
                    hands,
                    stats,
                    hand_stats,
                    lookup,
                    &new_board,
                    deck,
                    n_deck,
                    suit_counts,
                    cards_remaining - 1,
                    i + 1,
                    weight,
                );
                suit_counts[usize::from(suit)] -= 1;
            }
            i += 1;
        }
    }

    fn reserve_batch(&self, batch_size: u64) -> (u64, u64) {
        let total_batch_count = self.get_preflop_combo_count();
        let mut enum_pos = self.enum_pos.lock().unwrap();
        let start = *enum_pos;
        let end = std::cmp::min(total_batch_count, *enum_pos + batch_size);
        *enum_pos = end;
        (start, end)
    }

    fn get_preflop_combo_count(&self) -> u64 {
        let mut combo_count = 1u64;
        for c in &self.combined_ranges {
            combo_count *= c.size() as u64;
        }
        combo_count
    }

    fn get_postflop_combo_count(&self) -> u64 {
        let mut cards_in_deck = u64::from(CARD_COUNT);
        cards_in_deck -= u64::from(self.fixed_board.count());
        cards_in_deck -= self.dead_mask.count_ones() as u64;
        cards_in_deck -= 2 * self.n_players as u64;
        let board_cards_remaining = 5 - u64::from(self.fixed_board.count());
        let mut postflop_combos = 1u64;
        for i in 0..board_cards_remaining {
            postflop_combos *= cards_in_deck - i;
        }
        for i in 0..board_cards_remaining {
            postflop_combos /= i + 1;
        }
        postflop_combos
    }

    fn sim_random_walk_monte_carlo<R: Rng>(&self, rng: &mut R, tx: Sender<u8>) {
        let mut batch = SimulationResultsBatch::init(self.n_players);
        let mut hand_stats = vec![UnitResults::default(); self.n_players*NUM_HANDS];
        let card_dist: Uniform<u8> = Uniform::from(0..CARD_COUNT);
        let combo_dists: Vec<Uniform<usize>> = (0..self.combined_ranges.len())
            .into_iter()
            .map(|i| Uniform::from(0..self.combined_ranges[i].size()))
            .collect();
        let combined_range_dist = Uniform::from(0..self.combined_ranges.len());
        let mut used_cards_mask = 0u64;
        let mut player_hands = [Hand::default(); MAX_PLAYERS];
        let mut lookup = [0; MAX_PLAYERS];
        let mut combo_indexes = [0usize; MAX_PLAYERS];
        let mut combo_weights = [1u8; MAX_PLAYERS];
        let cards_remaining = 5 - self.fixed_board.count();

        if self.randomize_hole_cards(
            &mut used_cards_mask,
            &mut combo_indexes,
            &mut player_hands,
            &mut combo_weights,
            &mut lookup,
            rng,
            &combo_dists,
        ) {
            loop {
                let mut board = self.fixed_board;
                let mut weight = 1u64;
                for c in &combo_weights {
                    weight *= u64::from(*c);
                }
                randomize_board(
                    rng,
                    &mut board,
                    used_cards_mask | self.dead_mask,
                    cards_remaining,
                    &card_dist,
                );
                self.evaluate_hands(&player_hands, weight, &board, &lookup, &mut batch, &mut hand_stats, true);

                if (batch.eval_count & 0xfff) == 0 {
                    self.update_results(&tx, batch, hand_stats, false);
                    hand_stats = vec![UnitResults::default(); NUM_HANDS*self.n_players];
                    batch = SimulationResultsBatch::init(self.n_players);
                    if self.stopped.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                    if !self.randomize_hole_cards(
                        &mut used_cards_mask,
                        &mut combo_indexes,
                        &mut player_hands,
                        &mut combo_weights,
                        &mut lookup,
                        rng,
                        &combo_dists,
                    ) {
                        break;
                    }
                }

                let combined_range_idx = combined_range_dist.sample(rng);
                let combined_range = &self.combined_ranges[combined_range_idx];
                let mut combo_idx = combo_indexes[combined_range_idx];
                used_cards_mask -= combined_range.combos()[combo_idx].mask;
                let mut mask;
                loop {
                    if combo_idx == 0 {
                        combo_idx = combined_range.size();
                    }
                    combo_idx -= 1;
                    mask = combined_range.combos()[combo_idx].mask;
                    if (mask & used_cards_mask) == 0 {
                        break;
                    }
                }
                used_cards_mask |= mask;
                for i in 0..combined_range.player_count() {
                    let player_idx = combined_range.players()[i];
                    player_hands[player_idx] = combined_range.combos()[combo_idx].hands[i];
                    lookup[player_idx] = NUM_HANDS*player_idx+get_card_index(combined_range.combos()[combo_idx].hole_cards[i].0, combined_range.combos()[combo_idx].hole_cards[i].1);
                    combo_weights[player_idx] = combined_range.combos()[combo_idx].hole_cards[i].2;
                }
                combo_indexes[combined_range_idx] = combo_idx;
            }
        }
        self.update_results(&tx, batch, hand_stats, true);
    }

    fn randomize_hole_cards<R: Rng>(
        &self,
        used_cards_mask: &mut u64,
        combo_indexes: &mut [usize],
        player_hands: &mut [Hand],
        combo_weights: &mut [u8],
        lookup: &mut [usize;MAX_PLAYERS],
        rng: &mut R,
        combo_dists: &[Uniform<usize>],
    ) -> bool {
        let mut ok;
        for _ in 0..1000 {
            ok = true;
            *used_cards_mask = self.board_mask | self.dead_mask;
            for i in 0..self.combined_ranges.len() {
                let combo_idx = combo_dists[i].sample(rng);
                combo_indexes[i] = combo_idx;
                let combo = &self.combined_ranges[i].combos()[combo_idx];
                if (*used_cards_mask & combo.mask) != 0 {
                    ok = false;
                    break;
                }
                for j in 0..self.combined_ranges[i].player_count() {
                    let player_idx = self.combined_ranges[i].players()[j];
                    player_hands[player_idx] = combo.hands[j];
                    lookup[player_idx] = NUM_HANDS*player_idx+get_card_index(combo.hole_cards[j].0, combo.hole_cards[j].1);
                    combo_weights[player_idx] = combo.hole_cards[j].2;
                }
                *used_cards_mask |= combo.mask;
            }
            if ok {
                return true;
            }
        }
        false
    }

    fn update_results(&self, tx: &Sender<u8>, batch: SimulationResultsBatch, hand_batch: Vec<UnitResults>, finished: bool) {
        // get lock
        let mut results = self.results.write().unwrap();
        let mut batch_hands = 0u64;
        let mut batch_equity = 0f64;
        for (ri, bi) in results.hand_results.iter_mut().zip(hand_batch) {
            *ri += bi;
        }
        for i in 0..(1 << self.n_players) {
            let winner_count: u64 = u64::from((i as u32).count_ones());
            batch_hands += batch.wins_by_mask[i];
            let mut actual_player_mask = 0;
            for j in 0..self.n_players {
                if (i & (1 << j)) != 0 {
                    if winner_count == 1 {
                        results.wins[batch.player_ids[j]] += batch.wins_by_mask[i];
                        if batch.player_ids[j] == 0 {
                            batch_equity += batch.wins_by_mask[i] as f64;
                        }
                    } else {
                        results.ties[batch.player_ids[j]] +=
                            (batch.wins_by_mask[i] / winner_count) as f64;
                        if batch.player_ids[j] == 0 {
                            batch_equity += (batch.wins_by_mask[i] / winner_count) as f64;
                        }
                    }
                    actual_player_mask |= 1 << batch.player_ids[j];
                }
            }
            results.wins_by_mask[actual_player_mask] += batch.wins_by_mask[i];
        }
        batch_equity /= (batch_hands as f64) + 1e-9;

        results.eval_count += batch.eval_count;
        if !self.calc_exact {
            results.batch_sum += batch_equity;
            results.batch_sum2 += batch_equity * batch_equity;
            results.batch_count += 1.0;
            results.stdev = (1e-9 + results.batch_sum2
                - results.batch_sum * results.batch_sum / results.batch_count)
                .sqrt()
                / results.batch_count;

            let progress = (1.0 / (results.stdev / self.stdev_target).powi(2) * 100.0) as u8;
            tx.send(progress).unwrap();

            // calc variance
            if !finished && results.stdev < self.stdev_target {
                self.stopped.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        } else {
            let progress = (*self.enum_pos.lock().unwrap() as f64 / self.get_preflop_combo_count() as f64 * 100.0) as u8;
            tx.send(progress).unwrap();
        }
    }

    fn set_ranks(&self) {
        let mut results = self.results.write().unwrap();
        let board = Hand::from_bit_mask(self.board_mask);
        let get_draws = board.get_mask().count_ones() < 5;
        for (i, hand_range) in self.hand_ranges.iter().enumerate() {
            for hand in &hand_range.hands{
                let holding = Hand::from_hole_cards(hand.0, hand.1);
                let h = board + holding;
                let rank = h.evaluate();
                let mut category: u8 = (rank/HAND_CATEGORY_OFFSET) as u8;
                if category > 0 {
                    category -= 1;
                }
                results.hand_results[NUM_HANDS*i+get_card_index(hand.0, hand.1)].rank = category;
                let draw;
                if get_draws {
                    draw = get_draw(holding, board, category);
                } else {
                    draw = 0;
                }
                results.hand_results[NUM_HANDS*i+get_card_index(hand.0, hand.1)].draw = draw;
            }
        }
    }
    
    fn save_hand_weights(&self) -> u16 {
        let mut results = self.results.write().unwrap();
        let mut flag: u16 = 0;
        let mut check: bool = true;
        for (i, hand_range) in self.hand_ranges.iter().enumerate() {
            flag |= (1<<i);
            check = true;
            for hand in &hand_range.hands{
                let hand_index = NUM_HANDS*i+get_card_index(hand.0, hand.1);
                let hand_data = &mut results.hand_results[hand_index];
                hand_data.weight = hand.2 as u32;
                hand_data.valid = true;
                if check && hand_data.valid && hand_data.weight != 100 {
                    check = false;
                    flag ^= 1<<i;
                }
            }
            if check {
                for hand in &hand_range.hands{
                    let hand_index = NUM_HANDS*i+get_card_index(hand.0, hand.1);
                    let hand_data = &mut results.hand_results[hand_index];
                    if hand_data.valid {
                        hand_data.weight = 1;
                    }
                }
            }
        }
        return flag;
    }
    
    fn fix_hand_weights(&self, flag: u16) {
        let mut results = self.results.write().unwrap();
        for (i, hand_range) in self.hand_ranges.iter().enumerate() {
            let player_flag = flag & (1<<i) != 0;
            if player_flag {
                for hand in &hand_range.hands{
                    let hand_index = NUM_HANDS*i+get_card_index(hand.0, hand.1);
                    let hand_data = &mut results.hand_results[hand_index];
                    if hand_data.valid {
                        hand_data.weight = 100;
                    }
                }
            }
        }
        return flag;
    }

    #[inline(always)]
    fn evaluate_hands(
        &self,
        player_hands: &[Hand],
        weight: u64,
        board: &Hand,
        lookup: &[usize; MAX_PLAYERS],
        results: &mut SimulationResultsBatch,
        hand_results: &mut Vec<UnitResults>,
        flush_possible: bool,
    ) {
        // evaulate hands
        let mut winner_mask: u8 = 0;
        let mut best_score: u16 = 0;
        let mut player_mask: u8 = 1;
        let mut counter: usize = 0;
        let mut indexes: [usize; MAX_PLAYERS] = [0; MAX_PLAYERS];

        for i in 0..self.n_players {
            let hand: Hand = *board + player_hands[i];
            let score = if flush_possible {
                hand.evaluate()
            } else {
                hand.evaluate_without_flush()
            };
            match (score > best_score, score == best_score) {
                (true, false) => {
                    // add to wins by hand mask
                    best_score = score;
                    winner_mask = player_mask;
                    indexes[0] = i;
                    counter = 1;
                }
                (false, true) => {
                    winner_mask |= player_mask;
                    indexes[counter] = i;
                    counter += 1;
                }
                _ => {}
            }
            player_mask <<= 1;
        }

        for i in 0..self.n_players {
            hand_results[lookup[i]].update_total(weight);
        }
        
        if counter == 1 { //there is 1 winner
            hand_results[lookup[indexes[0]]].update_wins(weight);
        } else { //there is a tie
            for j in 0..counter {
                hand_results[lookup[indexes[j]]].update_ties(weight, counter);
            }
        }

        results.wins_by_mask[usize::from(winner_mask)] += weight;
        results.eval_count += 1;
    }
}

fn randomize_board<R: Rng>(
    rng: &mut R,
    board: &mut Hand,
    mut used_cards_mask: u64,
    cards_remaining: u32,
    card_dist: &Uniform<u8>,
) {
    // randomize board
    for _ in 0..cards_remaining {
        let mut card: u8;
        let mut card_mask: u64;
        loop {
            card = rng.sample(card_dist);
            card_mask = 1u64 << card;
            if (used_cards_mask & card_mask) == 0 {
                break;
            }
        }
        used_cards_mask |= card_mask;
        *board += CARDS[usize::from(card)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_range::{get_card_mask, HandRange};

    #[test]
    fn test_approx_weighted() {
        const ERROR: f64 = 0.01;
        const THREADS: u8 = 4;
        let (ranges, _) = HandRange::from_strings(["KK".to_string(), "AA@1,QQ".to_string()].to_vec());
        let equity = approx_equity(&ranges, 0, 0, THREADS, 0.001, Arc::new(AtomicBool::new(false)), |_u8|{}).unwrap();
        let equity = equity.equities;
        println!("{:?}", equity);
        assert!(equity[0] > 0.8130232455484216 - ERROR);
        assert!(equity[0] < 0.8130232455484216 + ERROR);
    }

    #[test]
    fn test_exact_weighted() {
        const THREADS: u8 = 8;
        let (ranges, _) = HandRange::from_strings(["KK".to_string(), "AA@1,QQ".to_string()].to_vec());
        let board_mask = get_card_mask("");
        let dead_mask = get_card_mask("");
        let equity = exact_equity(&ranges, board_mask, dead_mask, THREADS, Arc::new(AtomicBool::new(false)), |_u8|{}).unwrap();
        let equity = equity.equities;
        println!("{:?}", equity);
        assert_eq!(equity[0], 0.8130232455484216);
    }

    #[test]
    fn test_preflop_accuracy() {
        const THREADS: u8 = 8;
        let (ranges, _) = HandRange::from_strings(["AA".to_string(), "random".to_string()].to_vec());
        let board_mask = get_card_mask("");
        let dead_mask = get_card_mask("");
        let equity = exact_equity(&ranges, board_mask, dead_mask, THREADS, Arc::new(AtomicBool::new(false)), |_u8|{}).unwrap();
        let equity = equity.equities;
        println!("{:?}", equity);
        assert_eq!(equity[0], 0.8520371330210104);
    }
}
