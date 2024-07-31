use std::{
    fmt::Display,
    hash::{DefaultHasher, Hash, Hasher},
};

use blocks::{Block, BlockMode, RotatedState};
use macroquad::prelude::*;
use rand::gen_range;
use strum::IntoEnumIterator;

pub type Board = [[BlockState; 10]; 20];
pub const SIZE: f32 = 32.;

fn get_starting_pos() -> (f32, f32) {
    (screen_width() / 2. - 5. * SIZE, 100.)
}

fn board_drawer(board: &Board, preview_sequence: &mut [Block; 4]) {
    let (x, y) = get_starting_pos();
    for i in 0..board.len() {
        let row = board[i];
        for j in 0..row.len() {
            let block = board[i][j];
            let color = block.get_color();
            draw_rectangle(
                x + (j as f32) * SIZE,
                y + (i as f32) * SIZE,
                SIZE,
                SIZE,
                color,
            );
        }
    }
    for (u, block) in preview_sequence.iter().enumerate() {
        let color = BlockState::new_falling(block.tetromino, block.state).get_color();
        let board = block.get_minimizied_board();
        for i in 0..board.len() {
            for j in 0..board.len() {
                if board.get(i, j).is_falling() {
                    draw_rectangle(
                        x + ((15 + j) as f32) * SIZE,
                        y + ((u * 3 + 3 + i) as f32) * SIZE,
                        SIZE,
                        SIZE,
                        color,
                    );
                }
            }
        }
    }
}

fn random_block() -> Block {
    let mut hash = DefaultHasher::new();
    get_time().to_string().hash(&mut hash);
    let seed = hash.finish();
    macroquad::rand::srand(seed);
    let r = gen_range(0, blocks::BlockMode::iter().len());
    Block::new(blocks::BlockMode::iter().nth(r).unwrap(), RotatedState::Up)
}

fn falling_to_finished(board: &mut Board, preview_sequence: &mut [Block; 4]) -> Option<()> {
    board
        .iter_mut()
        .flat_map(|x| x.iter_mut())
        .filter(|x| x.is_falling())
        .for_each(|x| *x = BlockState::Occupied(OccupiedBlockStatus::Finished(x.get_color())));
    clear_lines(board);
    random_insert_block(board, preview_sequence)
}

fn random_insert_block(board: &mut Board, preview_sequence: &mut [Block; 4]) -> Option<()> {
    insert_block(board, preview_sequence[0])?;
    preview_sequence.rotate_left(1);
    if let Some(last) = preview_sequence.last_mut() {
        *last = random_block();
    }
    Some(())
}

fn clear_lines(board: &mut Board) {
    for u in board
        .iter_mut()
        .enumerate()
        .filter(|x| x.1.iter().all(|x| x.is_finished()))
        .map(|x| x.0)
        .collect::<Vec<_>>()
    {
        board[u] = [BlockState::Empty; 10];
        for i in (1..=u).rev() {
            board[i] = board[i - 1];
        }
    }
}

#[macroquad::main("Tetris")]
async fn main() {
    let mut board = [[BlockState::Empty; 10]; 20];

    let mut next_preview_piece: [Block; 4] = std::array::from_fn(|_| random_block());

    random_insert_block(&mut board, &mut next_preview_piece);
    let mut gravity_delta = get_time();
    let mut das_time = get_time();
    let mut arr_time = get_time();
    loop {
        let mut gravity_speed = 0.1;
        let dark_grey = Color::new(0.12, 0.17, 0.27, 1.);
        let grey = Color::new(0.25, 0.25, 0.32, 1.);
        clear_background(grey);
        let (x, y) = get_starting_pos();
        draw_rectangle(x, y, 10. * SIZE, 20. * SIZE, dark_grey);

        if board.last().unwrap().iter().any(|x| x.is_falling()) {
            if falling_to_finished(&mut board, &mut next_preview_piece).is_none() {
                break;
            }
        }

        let keys = macroquad::input::get_keys_pressed();
        if keys.contains(&KeyCode::Up) {
            rotate_block(&mut board);
        }

        if macroquad::input::get_keys_down().contains(&KeyCode::Down) {
            gravity_speed = 0.03;
        }

        let keys_direction = [
            (KeyCode::Left, Direction::Left),
            (KeyCode::Right, Direction::Right),
        ];

        // das_time = get_time();
        for (key, direction) in keys_direction {
            if keys.contains(&key) {
                das_time = get_time();
                if keys.contains(&KeyCode::Left) {
                    let _ = apply_movement(&mut board, Direction::Left);
                }
                if keys.contains(&KeyCode::Right) {
                    let _ = apply_movement(&mut board, Direction::Right);
                }
            }
            if (get_time() - das_time) > 0.3 {
                if macroquad::input::get_keys_down().contains(&key) {
                    if get_time() - arr_time > 0.1 {
                        let _ = apply_movement(&mut board, direction);
                        arr_time = get_time();
                    }
                }
            } else {
                arr_time = get_time() - 0.11;
            }
        }

        if get_time() - gravity_delta > gravity_speed {
            if apply_movement(&mut board, Direction::Down).is_err() {
                if falling_to_finished(&mut board, &mut next_preview_piece).is_none() {
                    break;
                }
            };
            gravity_delta = get_time();
        }

        board_drawer(&board, &mut next_preview_piece);
        next_frame().await;
    }
}

fn insert_block(board: &mut Board, block: Block) -> Option<()> {
    let small_block = block.get_minimizied_board();
    for (ur, row) in board
        .iter_mut()
        .take(small_block.len())
        .map(|x| {
            x.into_iter()
                .skip(5 - small_block.len() / 2)
                .take(small_block.len())
        })
        .enumerate()
    {
        for (uc, ock) in row.enumerate() {
            if ock.is_finished() {
                return None;
            }
            *ock = small_block.get(ur, uc);
        }
    }
    Some(())
}

fn rotate_block(board: &mut Board) {
    let (original_rotated_state, inner_block, board_len) = board
        .iter_mut()
        .flat_map(|x| x.iter_mut())
        .find_map(|x| match x {
            BlockState::Occupied(x) => match x {
                OccupiedBlockStatus::Falling(x) => {
                    let len = x.get_minimizied_board().len();
                    Some((x.state, x, len))
                }
                OccupiedBlockStatus::Finished(_) => None,
            },
            BlockState::Empty => None,
        })
        .unwrap();

    inner_block.rotate_right();

    let inner_block = *inner_block;

    let rotated_block = inner_block.get_minimizied_board();

    let mut rotater = move |col_sub, row_sub| {
        let col_skipping_count = board
            .iter()
            .skip_while(|x| !x.iter().any(|x| x.is_falling()))
            .take(board_len)
            .filter_map(|row| {
                row.iter()
                    .enumerate()
                    .find(|x| x.1.is_falling())
                    .map(|x| x.0)
            })
            .min();
        let row_skipping_count = board
            .iter()
            .position(|x| x.iter().any(|x| x.is_falling()))
            .and_then(|x| x.checked_sub(row_sub));
        let mut blocks_tobe_removed = Vec::new();
        for (x, (rx, row)) in board
            .iter()
            .enumerate()
            .skip(row_skipping_count.unwrap_or_default())
            .take(board_len)
            .map(|(r, row)| {
                (
                    r,
                    row.iter()
                        .enumerate()
                        .skip(
                            col_skipping_count
                                .and_then(|x| x.checked_sub(col_sub))
                                .unwrap_or_default(),
                        )
                        .take(board_len)
                        .enumerate(),
                )
            })
            .enumerate()
        {
            for (y, (ry, block)) in row {
                if block.is_finished() && rotated_block.get(x, y).is_falling() {
                    return;
                }
                if !block.is_finished() {
                    blocks_tobe_removed.push((rx, ry));
                }
            }
        }
        let mut rotated = Vec::new();
        for (x, (rx, row)) in board
            .iter_mut()
            .enumerate()
            .skip(row_skipping_count.unwrap_or_default())
            .take(board_len)
            .map(|(rx, row)| {
                (
                    rx,
                    row.iter_mut()
                        .enumerate()
                        .skip(
                            col_skipping_count
                                .and_then(|x| x.checked_sub(col_sub))
                                .unwrap_or_default(),
                        )
                        .take(board_len)
                        .enumerate(),
                )
            })
            .enumerate()
        {
            for (y, (ry, block)) in row {
                if (block.is_falling() || block.is_empty()) && rotated_block.get(x, y).is_falling()
                {
                    rotated.push((rx, ry));
                    *block = rotated_block.get(x, y);
                }
            }
        }
        blocks_tobe_removed.retain(|x| !rotated.contains(x));
        for (x, y) in blocks_tobe_removed {
            board[x][y] = BlockState::Empty;
        }
    };

    if matches!(inner_block.tetromino, BlockMode::I) {
        match original_rotated_state {
            RotatedState::Up => {
                rotater(0, 1);
            }
            RotatedState::Right => {
                rotater(2, 0);
            }
            RotatedState::Down => {
                rotater(0, 2);
            }
            RotatedState::Left => rotater(1, 0),
        };
    } else if matches!(inner_block.tetromino, BlockMode::O) {
        rotater(0, 0);
    } else {
        match original_rotated_state {
            RotatedState::Up => {
                rotater(0, 0);
            }
            RotatedState::Right => {
                rotater(1, 0);
            }
            RotatedState::Down => {
                rotater(0, 1);
            }
            RotatedState::Left => rotater(0, 0),
        };
    }
}

enum Direction {
    Left,
    Down,
    Right,
}

enum StopReason {
    NotMoving,
    Finished,
}

fn apply_movement(board: &mut Board, direction: Direction) -> Result<(), StopReason> {
    let board_view = board.clone();
    match direction {
        Direction::Left => {
            let iter = board_view
                .iter()
                .map(|x| x.iter().rev().skip_while(|x| !x.is_falling()).peekable());
            for mut x in iter {
                while let Some(y) = x.next() {
                    if y.is_falling() {
                        if let Some(peek) = x.peek().map(|x| x.is_finished()) {
                            if peek {
                                return Err(StopReason::NotMoving);
                            }
                        } else {
                            return Err(StopReason::NotMoving);
                        }
                    }
                }
            }
            for (ur, row) in board_view.iter().enumerate() {
                for (uc, block) in row.iter().enumerate() {
                    if block.is_falling() {
                        let Some(x) = board.get_mut(ur) else {
                            return Ok(());
                        };
                        let Some(x) = x.get_mut(uc - 1) else {
                            return Ok(());
                        };
                        *x = block.clone();
                        board[ur][uc] = BlockState::Empty;
                    }
                }
            }
        }
        Direction::Down => {
            let mut iter = board_view
                .iter()
                .skip_while(|x| !x.iter().any(|x| x.is_falling()))
                .peekable();
            while let Some(x) = iter.next() {
                for i in x
                    .iter()
                    .enumerate()
                    .filter(|x| x.1.is_falling())
                    .map(|x| x.0)
                {
                    if iter.peek().map(|x| x[i].is_finished()) == Some(true) {
                        return Err(StopReason::Finished);
                    }
                }
            }
            for (ur, row) in board_view.iter().enumerate().rev() {
                for (uc, block) in row.iter().enumerate() {
                    if block.is_falling() {
                        let Some(x) = board.get_mut(ur + 1) else {
                            return Ok(());
                        };
                        let Some(x) = x.get_mut(uc) else {
                            return Ok(());
                        };
                        *x = block.clone();
                        board[ur][uc] = BlockState::Empty;
                    }
                }
            }
        }
        Direction::Right => {
            let iter = board_view
                .iter()
                .map(|x| x.iter().skip_while(|x| !x.is_falling()).peekable());
            for mut x in iter {
                while let Some(y) = x.next() {
                    if y.is_falling() {
                        if let Some(peek) = x.peek().map(|x| x.is_finished()) {
                            if peek {
                                return Err(StopReason::NotMoving);
                            }
                        } else {
                            return Err(StopReason::NotMoving);
                        }
                    }
                }
            }
            for (ur, row) in board_view.iter().enumerate() {
                for (uc, block) in row.iter().enumerate().rev() {
                    if block.is_falling() {
                        let Some(x) = board.get_mut(ur) else {
                            return Ok(());
                        };
                        let Some(x) = x.get_mut(uc + 1) else {
                            return Ok(());
                        };
                        *x = block.clone();
                        board[ur][uc] = BlockState::Empty;
                    }
                }
            }
        }
    }
    Ok(())
}

mod blocks {
    use std::fmt::Display;

    use strum::EnumIter;

    use crate::BlockState;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum RotatedState {
        Up,
        Right,
        Down,
        Left,
    }

    impl Default for RotatedState {
        fn default() -> Self {
            Self::Up
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Block {
        pub tetromino: BlockMode,
        pub state: RotatedState,
    }

    impl Display for Block {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.tetromino.fmt(f)
        }
    }

    impl Block {
        pub const fn new(tetromino: BlockMode, state: RotatedState) -> Self {
            Self { tetromino, state }
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, EnumIter, derive_more::Display)]
    pub enum BlockMode {
        T,
        I,
        J,
        L,
        Z,
        S,
        O,
    }

    pub enum BoardType {
        Two([[BlockState; 2]; 2]),
        Three([[BlockState; 3]; 3]),
        Four([[BlockState; 4]; 4]),
    }

    impl BoardType {
        pub fn len(&self) -> usize {
            match self {
                BoardType::Two(_) => 2,
                BoardType::Three(_) => 3,
                BoardType::Four(_) => 4,
            }
        }
        pub fn get(&self, x: usize, y: usize) -> BlockState {
            match self {
                BoardType::Two(block) => block[x][y],
                BoardType::Three(block) => block[x][y],
                BoardType::Four(block) => block[x][y],
            }
        }
    }

    impl From<[[BlockState; 2]; 2]> for BoardType {
        fn from(v: [[BlockState; 2]; 2]) -> Self {
            Self::Two(v)
        }
    }

    impl From<[[BlockState; 4]; 4]> for BoardType {
        fn from(v: [[BlockState; 4]; 4]) -> Self {
            Self::Four(v)
        }
    }

    impl From<[[BlockState; 3]; 3]> for BoardType {
        fn from(v: [[BlockState; 3]; 3]) -> Self {
            Self::Three(v)
        }
    }

    impl Block {
        pub fn get_minimizied_board(&self) -> BoardType {
            match self.tetromino {
                BlockMode::T => {
                    use BlockState as S;
                    let make = S::new_falling(BlockMode::T, self.state);
                    match self.state {
                        RotatedState::Up => [
                            [S::Empty, make, S::Empty],
                            [make, make, make],
                            [S::Empty, S::Empty, S::Empty],
                        ],
                        RotatedState::Right => [
                            [S::Empty, make, S::Empty],
                            [S::Empty, make, make],
                            [S::Empty, make, S::Empty],
                        ],
                        RotatedState::Down => [
                            [S::Empty, S::Empty, S::Empty],
                            [make, make, make],
                            [S::Empty, make, S::Empty],
                        ],
                        RotatedState::Left => [
                            [S::Empty, make, S::Empty],
                            [make, make, S::Empty],
                            [S::Empty, make, S::Empty],
                        ],
                    }
                }
                .into(),
                BlockMode::I => {
                    use BlockState as S;
                    let make = S::new_falling(BlockMode::I, self.state);
                    match self.state {
                        RotatedState::Up => [
                            [S::Empty, S::Empty, S::Empty, S::Empty],
                            [make, make, make, make],
                            [S::Empty, S::Empty, S::Empty, S::Empty],
                            [S::Empty, S::Empty, S::Empty, S::Empty],
                        ],
                        RotatedState::Right => [
                            [S::Empty, S::Empty, make, S::Empty],
                            [S::Empty, S::Empty, make, S::Empty],
                            [S::Empty, S::Empty, make, S::Empty],
                            [S::Empty, S::Empty, make, S::Empty],
                        ],
                        RotatedState::Down => [
                            [S::Empty, S::Empty, S::Empty, S::Empty],
                            [S::Empty, S::Empty, S::Empty, S::Empty],
                            [make, make, make, make],
                            [S::Empty, S::Empty, S::Empty, S::Empty],
                        ],
                        RotatedState::Left => [
                            [S::Empty, make, S::Empty, S::Empty],
                            [S::Empty, make, S::Empty, S::Empty],
                            [S::Empty, make, S::Empty, S::Empty],
                            [S::Empty, make, S::Empty, S::Empty],
                        ],
                    }
                }
                .into(),
                BlockMode::J => {
                    use BlockState as S;
                    let make = S::new_falling(BlockMode::J, self.state);
                    let e = S::Empty;
                    match self.state {
                        RotatedState::Up => [[make, e, e], [make, make, make], [e, e, e]],
                        RotatedState::Right => [[e, make, make], [e, make, e], [e, make, e]],
                        RotatedState::Down => [[e, e, e], [make, make, make], [e, e, make]],
                        RotatedState::Left => [[e, make, e], [e, make, e], [make, make, e]],
                    }
                }
                .into(),
                BlockMode::L => {
                    use BlockState as S;
                    let make = S::new_falling(BlockMode::L, self.state);
                    let e = S::Empty;
                    match self.state {
                        RotatedState::Up => [[e, e, make], [make, make, make], [e, e, e]],
                        RotatedState::Right => [[e, make, e], [e, make, e], [e, make, make]],
                        RotatedState::Down => [[e, e, e], [make, make, make], [make, e, e]],
                        RotatedState::Left => [[make, make, e], [e, make, e], [e, make, e]],
                    }
                }
                .into(),
                BlockMode::Z => {
                    use BlockState as S;
                    let make = S::new_falling(BlockMode::Z, self.state);
                    let e = S::Empty;
                    match self.state {
                        RotatedState::Up => [[e, make, make], [make, make, e], [e, e, e]],
                        RotatedState::Right => [[e, make, e], [e, make, make], [e, e, make]],
                        RotatedState::Down => [[e, e, e], [e, make, make], [make, make, e]],
                        RotatedState::Left => [[make, e, e], [make, make, e], [e, make, e]],
                    }
                }
                .into(),
                BlockMode::S => {
                    use BlockState as S;
                    let make = S::new_falling(BlockMode::S, self.state);
                    let e = S::Empty;
                    match self.state {
                        RotatedState::Up => [[make, make, e], [e, make, make], [e, e, e]],
                        RotatedState::Right => [[e, e, make], [e, make, make], [e, make, e]],
                        RotatedState::Down => [[e, e, e], [make, make, e], [e, make, make]],
                        RotatedState::Left => [[e, make, e], [make, make, e], [make, e, e]],
                    }
                }
                .into(),
                BlockMode::O => {
                    use BlockState as S;
                    let make = S::new_falling(BlockMode::O, self.state);
                    match self.state {
                        RotatedState::Up => [[make; 2]; 2],
                        RotatedState::Right => [[make; 2]; 2],
                        RotatedState::Down => [[make; 2]; 2],
                        RotatedState::Left => [[make; 2]; 2],
                    }
                }
                .into(),
            }
        }
        pub fn rotate_right(&mut self) {
            let next = match self.state {
                RotatedState::Up => RotatedState::Right,
                RotatedState::Right => RotatedState::Down,
                RotatedState::Down => RotatedState::Left,
                RotatedState::Left => RotatedState::Up,
            };
            self.state = next;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockState {
    Occupied(OccupiedBlockStatus),
    Empty,
}

impl Display for BlockState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                BlockState::Occupied(x) => x.to_string(),
                BlockState::Empty => String::from("[ ]"),
            }
        )
    }
}

impl BlockState {
    pub const fn new_falling(s: BlockMode, state: RotatedState) -> Self {
        Self::Occupied(OccupiedBlockStatus::Falling(Block::new(s, state)))
    }
    pub fn is_occupied(&self) -> bool {
        matches!(self, Self::Occupied(_))
    }
    pub fn is_finished(&self) -> bool {
        match self {
            Self::Occupied(x) => x.is_finished(),
            _ => false,
        }
    }
    pub fn get_color(&self) -> Color {
        match self {
            BlockState::Occupied(x) => match x {
                OccupiedBlockStatus::Falling(x) => match x.tetromino {
                    BlockMode::T => PURPLE,
                    BlockMode::I => SKYBLUE,
                    BlockMode::J => ORANGE,
                    BlockMode::L => BLUE,
                    BlockMode::Z => GREEN,
                    BlockMode::S => RED,
                    BlockMode::O => YELLOW,
                },
                OccupiedBlockStatus::Finished(x) => *x,
            },
            BlockState::Empty => Color::new(0.41, 0.41, 0.50, 0.1),
        }
    }
    pub fn is_falling(&self) -> bool {
        match self {
            Self::Occupied(x) => x.is_falling(),
            _ => false,
        }
    }
    pub fn rotate_right(&mut self) {
        match self {
            BlockState::Occupied(x) => match x {
                OccupiedBlockStatus::Falling(x) => {
                    x.rotate_right();
                }
                OccupiedBlockStatus::Finished(_) => {}
            },
            BlockState::Empty => {}
        }
    }
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OccupiedBlockStatus {
    Falling(Block),
    Finished(Color),
}

impl OccupiedBlockStatus {
    /// Returns `true` if the occupied block status is [`Falling`].
    ///
    /// [`Falling`]: OccupiedBlockStatus::Falling
    #[must_use]
    pub fn is_falling(&self) -> bool {
        matches!(self, Self::Falling(..))
    }

    /// Returns `true` if the occupied block status is [`Finished`].
    ///
    /// [`Finished`]: OccupiedBlockStatus::Finished
    #[must_use]
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished(_))
    }
}

impl Display for OccupiedBlockStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Falling(x) => format!("[{}]", x.to_string()),
                Self::Finished(_) => String::from("[x]"),
            }
        )
    }
}
