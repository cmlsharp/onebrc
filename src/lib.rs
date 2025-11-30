#![feature(portable_simd)]
#![feature(cold_path)]

use std::{
    cmp, fmt, fs, hint,
    io::{self, prelude::*},
    ops::AddAssign,
    path::Path,
    simd::{LaneCount, SupportedLaneCount, prelude::*},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

mod hasher;
mod sso;

// station name: max is 100 bytes per the rules
// + 1 for ;
// + 5 for -xx.x
const MAX_LINE_LEN: usize = 100 + 1 + 5;

/// I use hashbrown::HashMap instead of the stl hashmap because it supports the
/// entry_ref API
/// ahash::AHasher is slower
//use hashbrown::HashMap;
type HashMap<K, V> = std::collections::HashMap<K, V, hasher::BuildFastHasher>;

// used to set initial capacities for hash tables
const INIT_CAPACITY: usize = 1024;

// these are the number of lanes for newlines and semicolons respectively that seem to minimize
// time
// they are different because semicolons are searched for _within_ a line
// so its usually a smaller window
const NL_LANES: usize = 16;
const SC_LANES: usize = 8;

// small string optimization byte string
pub type StationName = sso::SsoVec;

fn find_newline(buffer: &[u8]) -> Option<usize> {
    find_byte_simd::<NL_LANES>(b'\n', buffer)
}

fn find_semicolon(buffer: &[u8]) -> Option<usize> {
    find_byte_simd::<SC_LANES>(b';', buffer)
}

fn find_byte_simd<const LANES: usize>(byte: u8, mut buffer: &[u8]) -> Option<usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut i = 0;
    while let Some((chunk, rest)) = buffer.split_first_chunk() {
        let bytes = Simd::<u8, LANES>::from_array(*chunk);
        let mask = bytes.simd_eq(Simd::splat(byte));
        if let Some(set) = mask.first_set() {
            return Some(i + set);
        }
        i += LANES;
        buffer = rest;
    }
    hint::cold_path();
    buffer.iter().position(|&b| b == byte).map(|p| i + p)
}

/// An iterator over the lines of buf. Uses SIMD to find newlines
pub struct ByteLines<'a> {
    buf: &'a [u8],
}

impl<'a> ByteLines<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }
}

impl<'a> Iterator for ByteLines<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.is_empty() {
            return None;
        }

        let pos = find_newline(self.buf).unwrap_or(self.buf.len() - 1);
        let (line, new_buf) = unsafe { self.buf.split_at_unchecked(pos + 1) };
        self.buf = new_buf;
        Some(line)
    }
}

#[derive(Debug)]
pub struct Record {
    count: usize,
    sum: i32,
    min: i16,
    max: i16,
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

impl AddAssign<&'_ Record> for Record {
    fn add_assign(&mut self, rhs: &Record) {
        self.count += rhs.count;
        self.sum += rhs.sum;
        self.min = cmp::min(self.min, rhs.min);
        self.max = cmp::max(self.max, rhs.max);
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
    let frac_part = (unsafe { num.get_unchecked(dot_pos + 1) } - b'0') as i16;
    let abs = frac_part + int_part;

    hint::select_unpredictable(is_neg, -abs, abs)
}

/// Given a chunk returns an iterator over (station name, temperature * 10 as i16)
fn parse_chunk(chunk: &[u8]) -> impl Iterator<Item = (&[u8], i16)> {
    ByteLines::new(chunk).map(|line| unsafe {
        let pos = find_semicolon(line).unwrap_unchecked();
        let (station, temp_str) = line.split_at_unchecked(pos);
        (station, parse_temp_unchecked(temp_str))
    })
}

/// See docstring for worker function for the idea here
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
    const CHUNK_OVERLAP: usize = MAX_LINE_LEN;
    fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let file_len = file.metadata().unwrap().len();
        Ok(Self {
            file,
            file_len,
            num_chunks: file_len.div_ceil(Self::CHUNK_SIZE as u64) as usize,
            buffer: vec![0; Self::CHUNK_SIZE + Self::CHUNK_OVERLAP],
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

    // helper method
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

        let is_last_chunk = chunk_size < chunk.len();

        let end = if is_last_chunk {
            hint::cold_path();
            chunk_size
        } else {
            find_newline(&chunk[Self::CHUNK_SIZE..]).unwrap() + Self::CHUNK_SIZE + 1
        };

        &chunk[start..end]
    }
}

/// This is kind of an ultra simple work-stealing queue
/// Workers share an atomic counter of claimed chunks into the file. They 'claim' a chunk by
/// performing a fetch_add instruction on it, incrementing the counter. Then they begin the chunk
/// at chunk_start := CHUNK_SIZE * chunk_index. unfortunately chunks often start in the middle of a line
/// so instead of reading CHUNK_SIZE bytes, each worker actually reads CHUNK_SIZE + CHUNK_OVERLAP
/// bytes to ensure they can start and stop at newlines
///
/// Aside from edge cases of first and last chunk, the anatomy of a chunk looks like:
///
/// |_ _ _ _ \n _ _ _ _ _ _ _ \n _ ... _ _ \n _ _ _ _ \n _ _ _ _ _ |
/// ^           |                               |      |           |
/// |           |                               |      |           | - chunk_start + CHUNK_SIZE + CHUNK_OVERLAP
/// |           |                               |      |
/// |           |                               |      |
/// |           | - start reading               |      | - stop reading
/// |- chunk_start                              |- chunk_start + CHUNK_SIZE
///
/// Each worker accumulates a HashMap<Box<[u8]>, Record> which are then merged later
/// The nature of this challenge is that there aren't that many unique stations so boxing them
/// probably isn't that bad. If there were a lot, I'd consider arena allocation
fn accumulate_records(
    path: &Path,
    next_chunk: &AtomicU64,
) -> io::Result<HashMap<StationName, Record>> {
    // you would think this would be a classic use case for all threads to share a file descriptor
    // and use pread (via std::os::unix::fs::FileExt::read_exact_at), but it's faster to open a new
    // file descriptor in each thread. I guess it's a contention issue
    let mut records = HashMap::with_capacity_and_hasher(INIT_CAPACITY, Default::default());
    let mut reader = ChunkReader::new(path)?;

    // claims the next unclaimed chunk for this worker
    // I tried versions of this where workers claim N chunks at once
    // and it doesn't seem to make much of a difference, so contention doesn't seem to be an issue
    // (at least for current chunk sizes)
    let get_next = || next_chunk.fetch_add(1, Ordering::Relaxed) as usize;
    while let Some(chunk) = reader.read_chunk(get_next())? {
        for (station, temp) in parse_chunk(chunk) {
            match records.get_mut(station) {
                Some(stats) => {
                    *stats += temp;
                }
                None => {
                    // the double hash here is annoying
                    // but it shouldn't happen often
                    // hashbrown's entry_ref api is a nicer way to express this
                    records.insert(StationName::new(station), Record::from(temp));
                }
            };
        }
    }

    Ok(records)
}

pub fn process_file(path: &Path) -> io::Result<Vec<(StationName, Record)>> {
    let next_chunk = AtomicU64::new(0);

    let workers = std::thread::available_parallelism().unwrap().get();

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
    Ok(station_data)
}
