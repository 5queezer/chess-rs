use crate::board::{Board, move_to_str};

pub fn perft(b: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut nodes = 0;
    let mut moves = Vec::new();
    b.gen_moves(&mut moves);

    for m in moves {
        b.make_move(m);
        if !b.in_check(b.stm.flip()) {
            nodes += perft(b, depth - 1);
        }
        b.unmake();
    }

    nodes
}

pub fn test(b: &mut Board, depth: u32) {
    println!("perft {}", depth);
    let mut moves = Vec::new();
    b.gen_moves(&mut moves);
    let mut nodes = 0;
    for m in moves {
        b.make_move(m);
        if !b.in_check(b.stm.flip()) {
            let n = perft(b, depth - 1);
            println!("{} {}", move_to_str(m), n);
            nodes += n;
        }
        b.unmake();
    }
    println!("nodes {}", nodes);
}
