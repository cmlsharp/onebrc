use criterion::{Criterion, criterion_group, criterion_main};
use std::path::Path;

// Import your function - adjust the path based on your crate name
use onebrc::process_file;

fn benchmark_full_program(c: &mut Criterion) {
    let path = Path::new("measurements_big.txt"); // Your test file

    // Verify the file exists
    assert!(path.exists(), "Test file not found: {:?}", path);

    c.bench_function("process_file", |b| {
        b.iter(|| process_file(std::hint::black_box(path)).unwrap())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(15); // Fewer samples for slow benchmarks
    targets = benchmark_full_program
}
criterion_main!(benches);
