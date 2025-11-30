use onebrc::process_file;
use std::{
    io::{self, prelude::*},
    path::PathBuf,
};

fn main() -> io::Result<()> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("input data path must be provided"),
    );
    let start = std::time::Instant::now();
    let station_data = process_file(&path)?;
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    write!(stdout, "{{")?;
    for (k, v) in station_data {
        write!(stdout, "{}={v}, ", unsafe { k.as_str_unchecked() })?;
    }
    write!(stdout, "}}")?;
    stdout.flush()?;
    eprintln!("\n{:?}", start.elapsed());

    Ok(())
}
