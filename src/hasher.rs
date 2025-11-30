// This code is adapted from (and is a strictly less general version of) [foldhash](https://docs.rs/foldhash/latest/foldhash/).
use std::hash::Hasher;

#[derive(Default)]
pub struct FastHasher {
    accumulator: u64,
}

pub type BuildFastHasher = std::hash::BuildHasherDefault<FastHasher>;

#[inline(always)]
fn folded_multiply(x: u64, y: u64) -> u64 {
    // We compute the full u64 x u64 -> u128 product, this is a single mul
    // instruction on x86-64, one mul plus one mulhi on ARM64.
    let full = (x as u128).wrapping_mul(y as u128);
    let lo = full as u64;
    let hi = (full >> 64) as u64;

    // The middle bits of the full product fluctuate the most with small
    // changes in the input. This is the top bits of lo and the bottom bits
    // of hi. We can thus make the entire output fluctuate with small
    // changes to the input by XOR'ing these two halves.
    lo ^ hi
}

impl FastHasher {
    const FIXED_SEED: [u64; 2] = [0xc0ac29b7c97c50dd, 0x3f84d5b5b5470917];
    #[inline(always)]
    fn hash_bytes_short(bytes: &[u8], accumulator: u64) -> u64 {
        let len = bytes.len();
        let mut s0 = accumulator;
        let mut s1 = Self::FIXED_SEED[1];
        // XOR the input into s0, s1, then multiply and fold.
        if len >= 8 {
            s0 ^= u64::from_ne_bytes(bytes[0..8].try_into().unwrap());
            s1 ^= u64::from_ne_bytes(bytes[len - 8..].try_into().unwrap());
        } else if len >= 4 {
            s0 ^= u32::from_ne_bytes(bytes[0..4].try_into().unwrap()) as u64;
            s1 ^= u32::from_ne_bytes(bytes[len - 4..].try_into().unwrap()) as u64;
        } else if len > 0 {
            let lo = bytes[0];
            let mid = bytes[len / 2];
            let hi = bytes[len - 1];
            s0 ^= lo as u64;
            s1 ^= ((hi as u64) << 8) | mid as u64;
        }
        folded_multiply(s0, s1)
    }

    #[cold]
    #[inline(never)]
    // SAFETY: v.len() must be > 16
    unsafe fn hash_bytes_long(v: &[u8], accumulator: u64) -> u64 {
        debug_assert!(v.len() > 16);
        let mut s0 = accumulator;
        let mut s1 = s0.wrapping_add(Self::FIXED_SEED[1]);
        // for the purposes of this challenge, this can't happen
        if v.len() > 128 {
            unreachable!();
        }

        let len = v.len();
        unsafe {
            // SAFETY: our precondition ensures our length is at least 16, and the
            // above loops do not reduce the length under that. This protects our
            // first iteration of this loop, the further iterations are protected
            // directly by the checks on len.
            s0 = folded_multiply(load(v, 0) ^ s0, load(v, len - 16) ^ Self::FIXED_SEED[0]);
            s1 = folded_multiply(load(v, 8) ^ s1, load(v, len - 8) ^ Self::FIXED_SEED[0]);
            if len >= 32 {
                s0 = folded_multiply(load(v, 16) ^ s0, load(v, len - 32) ^ Self::FIXED_SEED[0]);
                s1 = folded_multiply(load(v, 24) ^ s1, load(v, len - 24) ^ Self::FIXED_SEED[0]);
                if len >= 64 {
                    s0 = folded_multiply(load(v, 32) ^ s0, load(v, len - 48) ^ Self::FIXED_SEED[0]);
                    s1 = folded_multiply(load(v, 40) ^ s1, load(v, len - 40) ^ Self::FIXED_SEED[0]);
                    if len >= 96 {
                        s0 = folded_multiply(
                            load(v, 48) ^ s0,
                            load(v, len - 64) ^ Self::FIXED_SEED[0],
                        );
                        s1 = folded_multiply(
                            load(v, 56) ^ s1,
                            load(v, len - 56) ^ Self::FIXED_SEED[0],
                        );
                    }
                }
            }
        }
        s0 ^ s1
    }
}

impl Hasher for FastHasher {
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        //self.accumulator = self.accumulator.rotate_right(len as u32);
        if bytes.len() <= 16 {
            self.accumulator = Self::hash_bytes_short(bytes, self.accumulator);
        } else {
            // SAFETY: we checked that len > 16
            unsafe { self.accumulator = Self::hash_bytes_long(bytes, self.accumulator) }
        }
    }

    #[inline(always)]
    fn finish(&self) -> u64 {
        self.accumulator
    }
}

/// Load 8 bytes into a u64 word at the given offset.
///
/// # Safety
/// You must ensure that offset + 8 <= bytes.len().
#[inline(always)]
unsafe fn load(bytes: &[u8], offset: usize) -> u64 {
    // In most (but not all) cases this unsafe code is not necessary to avoid
    // the bounds checks in the below code, but the register allocation became
    // worse if I replaced those calls which could be replaced with safe code.
    unsafe { bytes.as_ptr().add(offset).cast::<u64>().read_unaligned() }
}
