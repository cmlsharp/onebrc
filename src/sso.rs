use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    hint,
    mem::{ManuallyDrop, MaybeUninit},
};
const SMALL_SIZE: usize = std::mem::size_of::<AllocedSsoVec>();

#[cfg(target_endian = "big")]
compile_error!("this implementation assumes little endian");

#[repr(C)]
pub union SsoVec {
    local: LocalSsoVec,
    alloc: ManuallyDrop<AllocedSsoVec>,
}

impl SsoVec {
    #[inline]
    pub fn new(s: &[u8]) -> Self {
        debug_assert!(s.len() < usize::MAX / 2);
        if s.len() < SMALL_SIZE {
            Self {
                local: LocalSsoVec::new(s),
            }
        } else {
            hint::cold_path();
            Self {
                alloc: ManuallyDrop::new(AllocedSsoVec::new(s.to_vec().into_boxed_slice())),
            }
        }
    }

    fn is_local(&self) -> bool {
        ((unsafe { self.local.len }) >> 7) & 1 == 0
    }

    pub unsafe fn as_str_unchecked(&self) -> &str {
        unsafe { str::from_utf8_unchecked(self.as_ref()) }
    }
}

impl Drop for SsoVec {
    fn drop(&mut self) {
        if !self.is_local() {
            hint::cold_path();
            unsafe { ManuallyDrop::drop(&mut self.alloc) };
        }
    }
}

impl From<&[u8]> for SsoVec {
    fn from(value: &[u8]) -> Self {
        Self::new(value)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LocalSsoVec {
    data: MaybeUninit<[u8; SMALL_SIZE - 1]>,
    len: u8,
}

impl LocalSsoVec {
    fn new(data: &[u8]) -> Self {
        assert!(data.len() < SMALL_SIZE);
        let mut ret = Self {
            data: MaybeUninit::uninit(),
            len: 0,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ret.data.as_mut_ptr().cast(), data.len())
        };
        ret.len = data.len() as u8;
        ret
    }
    fn len(&self) -> usize {
        usize::from(self.len)
    }
}

impl AsRef<[u8]> for LocalSsoVec {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast(), self.len()) }
    }
}

#[repr(C)]
struct AllocedSsoVec {
    ptr: *mut u8,
    len: usize,
}

impl AllocedSsoVec {
    fn new(alloc: Box<[u8]>) -> Self {
        let ptr = Box::into_raw(alloc);
        Self {
            len: ptr.len() | (1 << (usize::BITS - 1)),
            ptr: ptr.cast(),
        }
    }
    fn len(&self) -> usize {
        self.len & !(1 << (usize::BITS - 1))
    }
}

impl AsRef<[u8]> for AllocedSsoVec {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len()) }
    }
}

unsafe impl Send for AllocedSsoVec {}

impl Drop for AllocedSsoVec {
    fn drop(&mut self) {
        let slice_ptr = std::ptr::slice_from_raw_parts_mut(self.ptr, self.len());
        drop(unsafe { Box::from_raw(slice_ptr) });
    }
}

impl AsRef<[u8]> for SsoVec {
    fn as_ref(&self) -> &[u8] {
        if self.is_local() {
            (unsafe { &self.local }).as_ref()
        } else {
            (unsafe { &self.alloc }).as_ref()
        }
    }
}

impl PartialEq for SsoVec {
    fn eq(&self, other: &Self) -> bool {
        self.is_local() == other.is_local() && self.as_ref() == other.as_ref()
    }
}

impl Eq for SsoVec {}

impl Hash for SsoVec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl Borrow<[u8]> for SsoVec {
    fn borrow(&self) -> &[u8] {
        self.as_ref()
    }
}
