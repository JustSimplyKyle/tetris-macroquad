use std::{
    fmt::Display,
    hash::{DefaultHasher, Hash, Hasher},
};

use blocks::{BlockMode, RotatedState};
use macroquad::prelude::*;
use rand::gen_range;
use strum::IntoEnumIterator;

pub type Board = [[BlockState; 10]; 20];
pub const SIZE: f32 = 32.;

fn get_starting_pos() -> (f32, f32) {
    (screen_width() / 2. - 5. * SIZE, 100.)
}

fn board_drawer(board: &Board, preview_sequence: &mut [BlockMode; 4]) {
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
        let color = BlockState::new_falling(*block).get_color();
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

fn random_block() -> BlockMode {
    let mut hash = DefaultHasher::new();
    get_time().to_string().hash(&mut hash);
    let seed = hash.finish();
    macroquad::rand::srand(seed);
    let r = gen_range(0, blocks::BlockMode::iter().len());
    blocks::BlockMode::iter().nth(r).unwrap()
}

fn falling_to_finished(board: &mut Board, preview_sequence: &mut [BlockMode; 4]) -> Option<()> {
    board
        .iter_mut()
        .flat_map(|x| x.iter_mut())
        .filter(|x| x.is_falling())
        .for_each(|x| *x = BlockState::Occupied(OccupiedBlockStatus::Finished(x.get_color())));
    clear_lines(board);
    random_insert_block(board, preview_sequence)
}

fn random_insert_block(board: &mut Board, preview_sequence: &mut [BlockMode; 4]) -> Option<()> {
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

    let mut next_preview_piece: [BlockMode; 4] = std::array::from_fn(|_| random_block());

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
                dbg!("DAS CHARGED!");
                if macroquad::input::get_keys_down().contains(&key) {
                    if get_time() - arr_time > 0.1 {
                        let _ = apply_movement(&mut board, direction);
                        arr_time = get_time();
                    }
                }
            } else {
                arr_time = get_time() - 0.11;
                dbg!("DAS CHARGING");
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

fn insert_block(board: &mut Board, block: BlockMode) -> Option<()> {
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
    let (rotated_state, mut inner_block, board_len) = board
        .iter()
        .flat_map(|x| x.iter())
        .find_map(|x| match x {
            BlockState::Occupied(x) => match x {
                OccupiedBlockStatus::Falling(x) => Some((
                    x.get_rotated_state(),
                    x.clone(),
                    x.get_minimizied_board().len(),
                )),
                OccupiedBlockStatus::Finished(_) => None,
            },
            BlockState::Empty => None,
        })
        .unwrap();
    inner_block.rotate_right();
    let rotated_block = inner_block.get_minimizied_board();

    let mut rotater = |col_sub, row_sub| {
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
        for (ur, row) in board
            .iter_mut()
            .skip(row_skipping_count.unwrap_or_default())
            .take(board_len)
            .map(|row| {
                row.iter_mut()
                    .skip(
                        col_skipping_count
                            .and_then(|x| x.checked_sub(col_sub))
                            .unwrap_or_default(),
                    )
                    .take(board_len)
                    .enumerate()
            })
            .enumerate()
        {
            for (uc, block) in row {
                if !block.is_finished() {
                    *block = rotated_block.get(ur, uc);
                }
            }
        }
    };

    if matches!(inner_block, BlockMode::I(_)) {
        match rotated_state {
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
    } else if matches!(inner_block, BlockMode::O(_)) {
        rotater(0, 0);
    } else {
        match rotated_state {
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

    #[derive(Copy, Clone, Debug, PartialEq, Eq, EnumIter)]
    pub enum BlockMode {
        T(T),
        I(I),
        J(J),
        L(L),
        Z(Z),
        S(S),
        O(O),
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

    impl BlockMode {
        pub fn get_minimizied_board(&self) -> BoardType {
            match self {
                BlockMode::T(x) => x.get_minimizied_board().into(),
                BlockMode::I(x) => x.get_minimizied_board().into(),
                BlockMode::J(x) => x.get_minimizied_board().into(),
                BlockMode::L(x) => x.get_minimizied_board().into(),
                BlockMode::Z(x) => x.get_minimizied_board().into(),
                BlockMode::S(x) => x.get_minimizied_board().into(),
                BlockMode::O(x) => x.get_minimizied_board().into(),
            }
        }
        pub fn get_rotated_state(&self) -> RotatedState {
            match self {
                Self::T(x) => x.state,
                Self::I(x) => x.state,
                Self::J(x) => x.state,
                Self::L(x) => x.state,
                Self::Z(x) => x.state,
                Self::S(x) => x.state,
                Self::O(x) => x.state,
            }
        }
        pub fn get_mut_rotated_state(&mut self) -> &mut RotatedState {
            match self {
                Self::T(x) => &mut x.state,
                Self::I(x) => &mut x.state,
                Self::J(x) => &mut x.state,
                Self::L(x) => &mut x.state,
                Self::Z(x) => &mut x.state,
                Self::S(x) => &mut x.state,
                Self::O(x) => &mut x.state,
            }
        }
        pub fn rotate_right(&mut self) {
            *self.get_mut_rotated_state() = match self.get_rotated_state() {
                RotatedState::Up => RotatedState::Right,
                RotatedState::Right => RotatedState::Down,
                RotatedState::Down => RotatedState::Left,
                RotatedState::Left => RotatedState::Up,
            };
        }
    }

    impl Display for BlockMode {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{}",
                match self {
                    Self::T(x) => x.to_string(),
                    Self::I(x) => x.to_string(),
                    Self::J(x) => x.to_string(),
                    Self::L(x) => x.to_string(),
                    Self::Z(x) => x.to_string(),
                    Self::S(x) => x.to_string(),
                    Self::O(x) => x.to_string(),
                }
            )
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct T {
        pub state: RotatedState,
    }

    impl Display for T {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "T")
        }
    }

    impl T {
        fn get_minimizied_board(&self) -> [[BlockState; 3]; 3] {
            use BlockState as S;
            let make = S::new_falling(BlockMode::T(*self));
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
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct I {
        pub state: RotatedState,
    }

    impl Display for I {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "I")
        }
    }

    impl I {
        fn get_minimizied_board(&self) -> [[BlockState; 4]; 4] {
            use BlockState as S;
            let make = S::new_falling(BlockMode::I(*self));
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
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct J {
        pub state: RotatedState,
    }

    impl Display for J {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "J")
        }
    }

    impl J {
        fn get_minimizied_board(&self) -> [[BlockState; 3]; 3] {
            use BlockState as S;
            let make = S::new_falling(BlockMode::J(*self));
            let e = S::Empty;
            match self.state {
                RotatedState::Up => [[make, e, e], [make, make, make], [e, e, e]],
                RotatedState::Right => [[e, make, make], [e, make, e], [e, make, e]],
                RotatedState::Down => [[e, e, e], [make, make, make], [e, e, make]],
                RotatedState::Left => [[e, make, e], [e, make, e], [make, make, e]],
            }
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct L {
        pub state: RotatedState,
    }

    impl Display for L {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "L")
        }
    }

    impl L {
        fn get_minimizied_board(&self) -> [[BlockState; 3]; 3] {
            use BlockState as S;
            let make = S::new_falling(BlockMode::L(*self));
            let e = S::Empty;
            match self.state {
                RotatedState::Up => [[e, e, make], [make, make, make], [e, e, e]],
                RotatedState::Right => [[e, make, e], [e, make, e], [e, make, make]],
                RotatedState::Down => [[e, e, e], [make, make, make], [make, e, e]],
                RotatedState::Left => [[make, make, e], [e, make, e], [e, make, e]],
            }
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct Z {
        pub state: RotatedState,
    }

    impl Display for Z {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Z")
        }
    }

    impl Z {
        fn get_minimizied_board(&self) -> [[BlockState; 3]; 3] {
            use BlockState as S;
            let make = S::new_falling(BlockMode::Z(*self));
            let e = S::Empty;
            match self.state {
                RotatedState::Up => [[e, make, make], [make, make, e], [e, e, e]],
                RotatedState::Right => [[e, make, e], [e, make, make], [e, e, make]],
                RotatedState::Down => [[e, e, e], [e, make, make], [make, make, e]],
                RotatedState::Left => [[make, e, e], [make, make, e], [e, make, e]],
            }
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct S {
        pub state: RotatedState,
    }

    impl Display for S {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "S")
        }
    }

    impl S {
        fn get_minimizied_board(&self) -> [[BlockState; 3]; 3] {
            use BlockState as S;
            let make = S::new_falling(BlockMode::S(*self));
            let e = S::Empty;
            match self.state {
                RotatedState::Up => [[make, make, e], [e, make, make], [e, e, e]],
                RotatedState::Right => [[e, e, make], [e, make, make], [e, make, e]],
                RotatedState::Down => [[e, e, e], [make, make, e], [e, make, make]],
                RotatedState::Left => [[e, make, e], [make, make, e], [make, e, e]],
            }
        }
    }
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct O {
        pub state: RotatedState,
    }

    impl Display for O {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "O")
        }
    }

    impl O {
        fn get_minimizied_board(&self) -> [[BlockState; 2]; 2] {
            use BlockState as S;
            let make = S::new_falling(BlockMode::O(*self));
            match self.state {
                RotatedState::Up => [[make; 2]; 2],
                RotatedState::Right => [[make; 2]; 2],
                RotatedState::Down => [[make; 2]; 2],
                RotatedState::Left => [[make; 2]; 2],
            }
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
    pub const fn new_falling(s: BlockMode) -> Self {
        Self::Occupied(OccupiedBlockStatus::Falling(s))
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
                OccupiedBlockStatus::Falling(x) => match x {
                    BlockMode::T(_) => PURPLE,
                    BlockMode::I(_) => SKYBLUE,
                    BlockMode::J(_) => ORANGE,
                    BlockMode::L(_) => BLUE,
                    BlockMode::Z(_) => GREEN,
                    BlockMode::S(_) => RED,
                    BlockMode::O(_) => YELLOW,
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
    Falling(BlockMode),
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
