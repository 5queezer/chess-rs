use crate::board::Board;

pub fn perft(b: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut moves = Vec::new();
    b.gen_moves(&mut moves);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0;
    for m in moves {
        b.make_move(m);
        nodes += perft(b, depth - 1);
        b.unmake();
    }
    nodes
}
