use crate::board::{move_to_str, Board, Move};
use rayon::prelude::*;

pub fn test(b: &Board, depth: u32) {
    println!("perft {}", depth);
    if depth == 0 {
        println!("nodes 1");
        return;
    }
    let mut moves = Vec::new();
    b.gen_moves(&mut moves);
    let use_parallel = depth >= 3 && moves.len() > 1;

    let results: Vec<(Move, u64)> = if use_parallel {
        moves
            .into_par_iter()
            .map(|m| {
                let n = perft_from_move(b, m, depth - 1);
                (m, n)
            })
            .collect()
    } else {
        moves
            .into_iter()
            .map(|m| {
                let n = perft_from_move(b, m, depth - 1);
                (m, n)
            })
            .collect()
    };

    let total_nodes: u64 = results.iter().map(|(_, n)| *n).sum();
    for (m, n) in results {
        println!("{} {}", move_to_str(m), n);
    }
    println!("nodes {}", total_nodes);
}

fn perft_from_move(b: &Board, m: Move, depth: u32) -> u64 {
    let mut child = b.clone();
    child.make_move(m);
    if child.in_check(child.stm.flip()) {
        return 0;
    }
    if depth == 0 {
        1
    } else {
        perft_inner(&mut child, depth)
    }
}

fn perft_inner(b: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut nodes = 0;
    let mut moves = Vec::new();
    b.gen_moves(&mut moves);

    for m in moves {
        b.make_move(m);
        if !b.in_check(b.stm.flip()) {
            nodes += perft_inner(b, depth - 1);
        }
        b.unmake();
    }

    nodes
}
