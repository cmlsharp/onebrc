#![feature(portable_simd)]
#![feature(cold_path)]
#![feature(hasher_prefixfree_extras)]

use std::{
    cmp, fmt, fs,
    hash::{BuildHasherDefault, Hasher},
    hint,
    io::{self, prelude::*},
    ops::AddAssign,
    path::Path,
    simd::prelude::*,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

mod sso;

// station name: max is 100 bytes per the rules
// + 1 for ;
// + 5 for -xx.x
const MAX_LINE_LEN: usize = 100 + 1 + 5;

type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FastHasher>>;

// used to set initial capacities for hash tables
const INIT_CAPACITY: usize = 1024;

// small string optimization byte string
pub type StationName = sso::SsoVec;

#[derive(Default)]
struct FastHasher {
    accumulator: u64,
}

impl FastHasher {
    const K: u64 = 0xf1357aea2e62a9c5;
    const SEED: u64 = 0x13198a2e03707344;
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.accumulator.rotate_left(26)
    }

    fn write_length_prefix(&mut self, _len: usize) {}

    fn write(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        // SAFETY: README promises station name lengths will be > 0
        unsafe { std::hint::assert_unchecked(len > 0) }

        let acc = Self::SEED
            ^ if len >= 4 {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                hint::cold_path();
                u32::from_le_bytes([bytes[0], bytes[len / 2], bytes[len - 1], 0])
            } as u64;

        self.accumulator = self.accumulator.wrapping_add(acc).wrapping_mul(Self::K);
    }
}

// SAFETY: buffer must contain a semicolon in the last 8 bytes
// (buffer may be < 8 bytes, but must contain a semicolon)
unsafe fn split_at_semicolon_unchecked(buffer: &[u8]) -> (&[u8], &[u8]) {
    const LANES: usize = 8;
    const SPLAT: Simd<u8, LANES> = Simd::splat(b';');

    let bytes = if let Some(chunk) = buffer.last_chunk() {
        Simd::<u8, LANES>::from_array(*chunk)
    } else {
        hint::cold_path();
        Simd::<u8, LANES>::load_or_default(buffer)
    };

    let set_pos = unsafe { bytes.simd_eq(SPLAT).first_set().unwrap_unchecked() };
    // there is no Mask::last_set, but we know there's only 1 ;
    let pos = buffer.len() - LANES + set_pos;
    let (before, after) = unsafe { buffer.split_at_unchecked(pos + 1) };
    (&before[..before.len() - 1], after)
}

fn find_newline(mut buffer: &[u8]) -> Option<usize> {
    const LANES: usize = 32;
    const SPLAT: Simd<u8, LANES> = Simd::splat(b'\n');

    let mut i = 0;
    while let Some((chunk, rest)) = buffer.split_first_chunk() {
        let bytes = Simd::<u8, LANES>::from_array(*chunk);
        let index = bytes.simd_eq(SPLAT).first_set().map(|set| set + i);
        if index.is_some() {
            return index;
        }
        i += LANES;
        buffer = rest;
    }

    let bytes = Simd::<u8, LANES>::load_or_default(buffer);
    let mask = bytes.simd_eq(SPLAT);
    mask.first_set().map(|set| i + set)
}

/// An iterator over the lines of buf. Uses SIMD to find newlines
struct ChunkParser<'a> {
    buf: &'a [u8],
}

impl<'a> ChunkParser<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }
}

impl<'a> Iterator for ChunkParser<'a> {
    type Item = (&'a [u8], i16);
    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.is_empty() {
            std::hint::cold_path();
            return None;
        }

        let pos = find_newline(self.buf).unwrap_or(self.buf.len() - 1);
        // SAFETY: pos is guaranteed to be < self.buf.len() so pos + 1 is guaranteed to be <=
        // self.buf.len()
        let (line, new_buf) = unsafe { self.buf.split_at_unchecked(pos + 1) };
        self.buf = new_buf;

        // SAFETY: README promises that every line contains a semicolon
        let (station, temp_str) = unsafe { split_at_semicolon_unchecked(line) };
        // SAFETY: pos is guaranteed to be < line.len()
        Some((station, unsafe { parse_temp_unchecked(temp_str) }))
    }
}

#[derive(Debug)]
pub struct Record {
    count: usize,
    sum: i32,
    min: i16,
    max: i16,
}
impl AddAssign<&Record> for Record {
    fn add_assign(&mut self, rhs: &Record) {
        self.count += rhs.count;
        self.sum += rhs.sum;
        // these are sufficiently rare, that the occasional branch misses are actually worth it
        // compared to a cmov instruction every time
        if rhs.max > self.max {
            std::hint::cold_path();
            self.max = rhs.max;
        }

        if rhs.min < self.min {
            std::hint::cold_path();
            self.min = rhs.min;
        }
    }
}

impl fmt::Display for Record {
    // weird output format demanded by the challenge
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1}/{:.1}/{:.1}",
            self.min as f64 / 10.,
            (self.sum as f64 / 10.) / self.count as f64,
            self.max as f64 / 10.
        )
    }
}

impl From<i16> for Record {
    fn from(value: i16) -> Self {
        Self {
            count: 1,
            sum: i32::from(value),
            min: value,
            max: value,
        }
    }
}

impl AddAssign<i16> for Record {
    fn add_assign(&mut self, rhs: i16) {
        *self += &Record::from(rhs);
    }
}

/// This parses floats of the form -?[0-9]?[0-9]\.[0-9]
/// it returns an integer which is the floating point number multiplied by 10
/// (since the float is guaranteed to only have one digit after the decimal)
/// integer arithmetic is cheaper than floating point arithmetic.
// SAFETY: requires line be at least 3 characters (4 if it contains two digits before '.')
unsafe fn parse_temp_unchecked(bytes: &[u8]) -> i16 {
    unsafe { hint::assert_unchecked(bytes.len() >= 3) }
    let is_neg = bytes[0] == b'-';

    let num = &bytes[is_neg as usize..];

    let first_digit = (num[0] - b'0') as i16;
    let has_two_digits = (num[1] != b'.') as usize;
    let second_digit = (num[1].wrapping_sub(b'0')) as i16;

    // Single digit: first * 10 + frac
    // Two digits: (first * 10 + second) * 10 + frac = first * 100 + second * 10 + frac
    let int_part = first_digit * (10 + 90 * has_two_digits as i16)
        + second_digit * (10 * has_two_digits as i16);

    let dot_pos = 1 + has_two_digits;
    // compiler can't elide this bounds check without a hint for some reason, but this is safe
    // Could do get_unchecked here?
    let frac_part = (unsafe { num.get_unchecked(dot_pos + 1) }.wrapping_sub(b'0')) as i16;
    let abs = frac_part + int_part;

    hint::select_unpredictable(is_neg, -abs, abs)
}

/// Given a chunk returns an iterator over (station name, temperature * 10 as i16)
struct ChunkReader {
    file: fs::File,
    file_len: u64,
    num_chunks: usize,
    buffer: Vec<u8>,
}

impl ChunkReader {
    // size of chunk read from file.
    // generally in the range of what seems to be optimal
    const CHUNK_SIZE: usize = 1 << 16;
    // Needs to be at least as large as the longest line in the input file
    const OVERLAP: usize = MAX_LINE_LEN;
    fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let file_len = file.metadata().unwrap().len();
        Ok(Self {
            file,
            file_len,
            num_chunks: file_len.div_ceil(Self::CHUNK_SIZE as u64) as usize,
            buffer: vec![0; Self::CHUNK_SIZE + Self::OVERLAP],
        })
    }

    /// Reads the `chunk_index`th chunk from self.file
    fn read_chunk(&mut self, chunk_index: usize) -> io::Result<Option<&[u8]>> {
        if chunk_index >= self.num_chunks {
            return Ok(None);
        }
        let chunk_start = chunk_index as u64 * (Self::CHUNK_SIZE as u64);
        let chunk_size = cmp::min(self.buffer.len(), (self.file_len - chunk_start) as usize);

        self.file.seek(io::SeekFrom::Start(chunk_start))?;
        self.file.read_exact(&mut self.buffer[..chunk_size])?;
        Ok(Some(Self::trim_chunk(
            &self.buffer,
            chunk_index == 0,
            chunk_size,
        )))
    }

    // extracts chunk from self.buffer
    // for chunks after the first one, we seek to the first newline
    // for chunks besides the last one, we read forward after CHUNK_SIZE until
    // we find a newline
    // This contains plenty of branches but they're very predictable
    fn trim_chunk(chunk: &[u8], is_first_chunk: bool, chunk_size: usize) -> &[u8] {
        let start = if is_first_chunk {
            hint::cold_path();
            0
        } else {
            find_newline(chunk).unwrap() + 1
        };

        let end = if chunk_size < chunk.len() {
            // this occurs when this is the last chunk and chunk size doesn't divide file size
            hint::cold_path();
            chunk_size
        } else {
            find_newline(&chunk[Self::CHUNK_SIZE..]).unwrap() + Self::CHUNK_SIZE + 1
        };

        unsafe { chunk.get_unchecked(start..end) }
    }
}

fn insert_record(station: &[u8], temp: i16, records: &mut HashMap<StationName, Record>) {
    match records.get_mut(station) {
        Some(stats) => {
            *stats += temp;
        }
        None => {
            // the double hash here is annoying
            // but it shouldn't happen often
            // without an entry_ref api the alternative is entry()
            // which takes keys by value, hence requiring allocations even in the happy path
            records.insert(StationName::new(station), Record::from(temp));
        }
    };
}

fn accumulate_records(
    path: &Path,
    next_chunk: &AtomicU64,
) -> io::Result<HashMap<StationName, Record>> {
    let mut records = HashMap::with_capacity_and_hasher(INIT_CAPACITY, Default::default());
    let mut reader = ChunkReader::new(path)?;

    // claims the next unclaimed chunk for this worker
    // I tried versions of this where workers claim N chunks at once
    // and it doesn't seem to make much of a difference, so contention doesn't seem to be an issue
    // (at least for current chunk sizes)
    let get_next = || next_chunk.fetch_add(1, Ordering::Relaxed) as usize;
    while let Some(chunk) = reader.read_chunk(get_next())? {
        for (station, temp) in ChunkParser::new(chunk) {
            insert_record(station, temp, &mut records)
        }
    }

    Ok(records)
}

pub fn process_file(path: &Path) -> io::Result<Vec<(StationName, Record)>> {
    let next_chunk = AtomicU64::new(0);

    let workers = std::env::var("ONEBRC_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get());

    let mut station_data = std::thread::scope(|scope| {
        let (tx, rx) = mpsc::sync_channel(workers);
        for _ in 0..workers {
            scope.spawn({
                let tx = tx.clone();
                let next_chunk = &next_chunk;
                move || tx.send(accumulate_records(path, next_chunk))
            });
        }
        drop(tx);
        rx.into_iter().try_fold(
            HashMap::with_capacity_and_hasher(INIT_CAPACITY, Default::default()),
            |mut acc, map| {
                map?.into_iter().for_each(|(station, record)| {
                    acc.entry(station)
                        .and_modify(|rec| *rec += &record)
                        .or_insert(record);
                });
                io::Result::Ok(acc)
            },
        )
    })?
    .into_iter()
    .collect::<Vec<_>>();
    station_data.sort_unstable_by(|(k1, _), (k2, _)| unsafe {
        k1.as_str_unchecked().cmp(k2.as_str_unchecked())
    });
    println!("{:?}", COUNTER.load(Ordering::Relaxed));
    Ok(station_data)
}

pub fn print(data: &[(StationName, Record)]) -> io::Result<()> {
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    write!(stdout, "{{")?;
    for (k, v) in data {
        write!(stdout, "{}={v}, ", unsafe { k.as_str_unchecked() })?;
    }
    write!(stdout, "}}")?;
    stdout.flush()?;
    Ok(())
}
