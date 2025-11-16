use crate::board::{move_to_str, Board, Move};
use rayon::prelude::*;

pub fn test(board: &Board, depth: u32) {
    println!("perft {}", depth);

    if depth == 0 {
        println!("nodes 1");
        return;
    }

    let mut moves = Vec::new();
    board.gen_moves(&mut moves);

    let use_parallel = depth >= 3 && moves.len() > 1;

    let results: Vec<(Move, u64)> = if use_parallel {
        moves.into_par_iter().map(|m| (m, count_nodes_from_move(board, m, depth - 1))).collect()
    } else {
        moves.into_iter().map(|m| (m, count_nodes_from_move(board, m, depth - 1))).collect()
    };

    let total_nodes: u64 = results.iter().map(|(_, n)| *n).sum();

    for (m, n) in results {
        println!("{} {}", move_to_str(m), n);
    }
    println!("nodes {}", total_nodes);
}

fn count_nodes_from_move(board: &Board, m: Move, depth: u32) -> u64 {
    let mut child = board.clone();
    child.make_move(m);

    if child.in_check(child.stm.flip()) {
        return 0;
    }

    if depth == 0 {
        1
    } else {
        count_nodes(&mut child, depth)
    }
}

fn count_nodes(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut nodes = 0;
    let mut moves = Vec::new();
    board.gen_moves(&mut moves);

    for m in moves {
        board.make_move(m);
        if !board.in_check(board.stm.flip()) {
            nodes += count_nodes(board, depth - 1);
        }
        board.unmake();
    }

    nodes
}
