use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use rand::Rng;
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use fibo_air::fibonacci_bn254_test;

const N: usize = 5;

// Returns (number_steps, final_value)
fn fibonacci(log_n: usize) -> (usize, u32) {
    let steps: usize = 1 << log_n;

    let mut a: u32 = 0;
    let mut b: u32 = 1;

    for _ in 0..steps {
        let next = a + b;
        a = b;
        b = next;
    }

    (steps, a)
}

fn benchmark_fibonacci(c: &mut Criterion) {
    let mut vec = Vec::new();
    for log_n in 1..=N {
        let (num_steps, res) = fibonacci(log_n);
        println!("{} {} {}", log_n, num_steps, res);
        vec.push((log_n, num_steps, res));
    }

    c.bench_function(&format!("fibonacci bls12-337/{}", N), |b| {
        b.iter(|| {
            for (log_n, n, res) in vec.clone() {
                fibonacci_bn254_test(log_n, n, res);
            }
        })
    });
}

criterion_group!(benches, benchmark_fibonacci);
criterion_main!(benches);