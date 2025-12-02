use std::{io, path::PathBuf};

fn main() -> io::Result<()> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("input data path must be provided"),
    );
    let start = std::time::Instant::now();
    onebrc::print(&onebrc::process_file(&path)?)?;
    eprintln!("\n{:?}", start.elapsed());

    Ok(())
}
