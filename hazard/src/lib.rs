/// Hazard pointers is a safe reclamation method. It protects objects
/// from being reclaimed while being accessed by one or more threads, but
/// allows objects to be removed concurrently while being accessed.
///
/// What is a Hazard Pointer?
/// -------------------------
/// A hazard pointer is a single-writer multi-reader pointer that can
/// be owned by at most one thread at a time. To protect an object A
/// from being reclaimed while in use, a thread X sets one of its
/// owned hazard pointers, P, to the address of A. If P is set to &A
/// before A is removed (i.e., it becomes unreachable) then A will not be
/// reclaimed as long as P continues to hold the value &A.
///
/// Why use hazard pointers?
/// ------------------------
/// - Speed and scalability.
/// - Can be used while blocking.
///
/// When not to use hazard pointers?
/// --------------------------------
/// - When thread local data is not supported efficiently.
///
/// Basic Interface
/// ---------------
/// - In the hazptr library, raw hazard pointers are not exposed to
///   users. Instead, each instance of the class hazptr_holder owns
///   and manages at most one hazard pointer.
/// - Typically classes of objects protected by hazard pointers are
///   derived from a class template hazptr_obj_base that provides a
///   member function retire(). When an object A is removed,
///   A.retire() is called to pass responsibility for reclaiming A to
///   the hazptr library. A will be reclaimed only after it is not
///   protected by hazard pointers.
/// - The essential components of the hazptr API are:
///   o hazptr_holder: Class that owns and manages a hazard pointer.
///   o protect: Member function of hazptr_holder. Protects
///     an object pointed to by an atomic source (if not null).
///       T* protect(const atomic<T*>& src);
///   o hazptr_obj_base<T>: Base class for protected objects.
///   o retire: Member function of hazptr_obj_base that automatically
///     reclaims the object when safe.
///       void retire();
use std::{collections::HashSet, sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering}};

use lazy_static::lazy_static;

fn debug(s: &str) {
    println!("{}", s);
}

pub trait Deleter {
    fn delete(&self);
}

#[derive(Default)]
struct DummyDeleter {}

impl Deleter for DummyDeleter {
    fn delete(&self) {
        
    }
}

lazy_static! {
    static ref SHARED_DOMAIN: HazPtrDomain = HazPtrDomain::new();
    static ref DUMMY_DELETER: DummyDeleter = DummyDeleter::default();
}

// Base class for protected objects.
// hazptr_obj_base in folly.
trait HazPtrObjBase {
    fn domain(&self) -> &'static HazPtrDomain;

    fn deleter(&self) -> &'static dyn Deleter;

    /// # Safety
    /// 1. No new reader can access the retired `ptr`.
    /// 2. The pointer can be safely deleted by `deleter()`.
    /// NOTE: Other readers can still access the `ptr` which the retired HazPtr used to point to.
    // Put myself into domain's retire list, providing a Drop func.
    fn retire(&mut self) {
        let deleter = self.deleter();
        let domain = self.domain().clone();
        domain.retire(self as *mut Self as *mut u8, deleter);
    }

    // fn retire_by(self: *mut Self, deleter: &'static dyn Deleter) {
    //     self.domain().retire(self as *mut Self as *mut u8, deleter);
    // }
}

/// Every thread creates its own `HazPtr` with the same ptr.
// hazptr_obj in folly.
#[derive(Debug)]
pub struct HazPtr {
    ptr: AtomicPtr<u8>,
    next: AtomicPtr<HazPtr>,
    active: AtomicBool,
}

impl HazPtr {
    fn new() -> Self {
        HazPtr {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicPtr::new(std::ptr::null_mut()),
            active: AtomicBool::new(true),
        }
    }

    fn protect(&self, ptr: *mut u8) {
        self.ptr.store(ptr, Ordering::SeqCst);
    }
}


/// Class for automatic acquisition and release of hazard pointers,
/// and interface for hazard pointer operations.
/// It guard the `ptr` during its lifetime.
/// NOTE: reset function will reset the HazPtr, yields current guards,
/// and make it reusable later.
#[derive(Default)]
pub struct HazPtrHolder {
    // Either empty, or some ptr that is being protected.
    inner: Option<&'static HazPtr>,
}

impl HazPtrHolder {
    // Get a hazptr.
    fn hazptr(&mut self) -> &HazPtr {
        if let Some(p) = self.inner {
            debug("has hazptr");
            // We can reused.
            p
        } else {
            // Allocate a hazptr
            debug("acquire hazptr");
            let p = SHARED_DOMAIN.acquire();
            debug(&format!("acquired hazptr {:?}", p as *const _ as *const u8));
            self.inner = Some(p);
            p
        }
    }

    /// Protects an object pointed to by an atomic source (if not null).
    ///  T* protect(const atomic<T*>& src);
    pub fn load<T>(&mut self, ptr: &AtomicPtr<T>) -> &T {
        let hazptr = self.hazptr();
        debug("got hazptr");
        // Check the ptr is not mutated during the peotecting.
        let mut before = ptr.load(Ordering::SeqCst);
        loop {
            // AtomicPtr has the same memory layout as `*mut T`.
            hazptr.protect(before as *mut u8);
            let after = ptr.load(Ordering::SeqCst);
            if before == after {
                assert_ne!(after, std::ptr::null_mut());
                return unsafe { &*after };
            }
            before = after;
        }
    }

    pub fn reset(&mut self) {
        if let Some(hazptr) = self.inner {
            hazptr.ptr.store(std::ptr::null_mut(), Ordering::SeqCst);
        }
    }
}

impl Drop for HazPtrHolder {
    fn drop(&mut self) {
        debug("drop holder");
        self.reset();

        if let Some(hazptr) = self.inner {
            hazptr.active.store(false, Ordering::SeqCst);
        }
    }
}

struct HazPtrList {
    head: AtomicPtr<HazPtr>,
}

impl HazPtrList {
    fn new() -> Self {
        HazPtrList {
            head: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

struct Retired {
    // There will be `HazPtr`s for diffferent `ptr`s in retired list.
    ptr: *mut u8,
    deleter: &'static dyn Deleter,
    next: AtomicPtr<Retired>,
}

impl Retired {
    fn new(ptr: *mut u8, deleter: &'static dyn Deleter) -> Self {
        Retired {
            ptr,
            deleter,
            next: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

struct RetiredList {
    head: AtomicPtr<Retired>,
    count: AtomicUsize,
}

impl RetiredList {
    fn new() -> Self {
        RetiredList {
            head: AtomicPtr::new(std::ptr::null_mut()),
            count: AtomicUsize::new(0),
        }
    }
}

/// Holds linked list of HazPtrs.
pub struct HazPtrDomain {
    hazptrs: HazPtrList,
    retired: RetiredList,
}

impl HazPtrDomain {
    fn new() -> Self {
        HazPtrDomain {
            hazptrs: HazPtrList::new(),
            retired: RetiredList::new(),
        }
    }

    fn maybe_gc(&self) -> usize {
        self.gc(false)
    }

    fn activate_count(&self) -> usize {
        let mut l = self.hazptrs.head.load(Ordering::SeqCst);
        let mut debug_count = 0;
        while !l.is_null() {
            let r = unsafe {&*l};
            if r.active.load(Ordering::SeqCst) {
                debug_count += 1;
            }
            l = r.next.load(Ordering::SeqCst);
        }
        debug_count
    }

    fn retired_count(&self) -> usize {
        let mut l = self.retired.head.load(Ordering::SeqCst);
        let mut debug_count = 0;
        while !l.is_null() {
            let r = unsafe {&*l};
            debug_count += 1;
            l = r.next.load(Ordering::SeqCst);
        }
        debug_count
    }

    // We must make sure there is no attaching reader before reclaiming(gc) the pointer.
    fn gc(&self, block: bool) -> usize {
        // We will address `count` later.
        let old_head = self.retired.head.swap(std::ptr::null_mut(), Ordering::SeqCst);
        if old_head.is_null() {
            return 0;
        }

        let mut l = self.hazptrs.head.load(Ordering::SeqCst);
        let mut remaining_readers = HashSet::new();
        while !l.is_null() {
            let r = unsafe {&*l};
            if r.active.load(Ordering::SeqCst) {
                remaining_readers.insert(r.ptr.load(Ordering::SeqCst));
            }
            l = r.next.load(Ordering::SeqCst);
        }

        // There is no ABA problem because a object at the same address must not be allocated
        // becore the old one is reclaimed.
        let mut reclaimed = 0usize;
        let mut l = old_head.clone();
        let dummy = Box::into_raw(Box::new(Retired::new(std::ptr::null_mut(), &*DUMMY_DELETER as &'static dyn Deleter)));
        let new_head = dummy;
        let mut tail = dummy;
        while !l.is_null() {
            let r = unsafe {&*l};
            let next = r.next.load(Ordering::SeqCst);
            if !remaining_readers.contains(&r.ptr) {
                debug(&format!("gced {:?}", r.ptr));
                r.deleter.delete();
                reclaimed += 1;
            } else {
                debug("has reader");
                // TODO Can we merge Retired which points to the same ptr?
                unsafe {&* tail}.next.store(l, Ordering::SeqCst);
                tail = l;
            }
            l = next;
        }
        if tail != dummy {
            // If there is something remaining.
            let mut cur_head = self.retired.head.load(Ordering::SeqCst);
            let real_new_head = unsafe {&*new_head}.next.load(Ordering::SeqCst);
            loop {
                unsafe {&* tail}.next.store(cur_head, Ordering::SeqCst);
                match self.retired.head.compare_exchange_weak(
                    cur_head,
                    real_new_head,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        break;
                    }
                    Err(head) => {
                        cur_head = head;
                    }
                }
            }
        }
        self.retired.count.fetch_sub(reclaimed, Ordering::SeqCst);
        if tail != dummy && block {
            // Caller wants to reclaim _everything_, but some were left, so try again.
            std::thread::yield_now();
            // NOTE: Allows tail recursion by passing down reclaimed
            return reclaimed + self.gc(true);
        }
        debug(&format!("gc finish {} is_block {}", reclaimed, block));
        reclaimed
    }

    fn retire(&self, ptr: *mut u8, deleter: &'static dyn Deleter) {
        debug("retired");
        self.retired.count.fetch_add(1, Ordering::SeqCst);
        let mut head_ptr = self.retired.head.load(Ordering::SeqCst);

        loop {
            let r = Retired::new(ptr, deleter);
            r.next.store(head_ptr, Ordering::SeqCst);
            let rp = Box::into_raw(Box::new(r));
            match self.retired.head.compare_exchange_weak(
                head_ptr,
                rp,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    break;
                }
                Err(head) => {
                    head_ptr = head;
                }
            }
        }

        if self.retired.count.load(Ordering::SeqCst) != 0 {
            self.maybe_gc();
        }
    }

    fn acquire(&self) -> &HazPtr {
        let mut head_ptr = self.hazptrs.head.load(Ordering::SeqCst);
        let mut cur = head_ptr.clone();
        loop {
            // Iterate over all list to find something to reuse.
            while !cur.is_null() && unsafe { &*cur }.active.load(Ordering::SeqCst) {
                cur = unsafe { &*cur }.next.load(Ordering::SeqCst);
            }
            if cur.is_null() {
                debug("aq by allocate");
                // Should allocate a new node.
                let nnb: *mut HazPtr = Box::into_raw(Box::new(HazPtr::new()));
                let safe_nnb = unsafe { &*nnb };
                safe_nnb.next.store(head_ptr, Ordering::SeqCst);
                match self.hazptrs.head.compare_exchange_weak(
                    head_ptr,
                    nnb,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        return safe_nnb;
                    }
                    Err(node) => {
                        head_ptr = node;
                    }
                }
            } else {
                debug("aq by reuse");
                let safe_cur = unsafe { &*cur };
                if safe_cur
                    .active
                    .compare_exchange_weak(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    return safe_cur;
                }
                // Keep retrying otherwise.
            }
        }
    }
}

struct HazPtrObj<T> {
    inner: T,
    domain: &'static HazPtrDomain,
}

impl<T> HazPtrObj<T> {
    fn new_with_shared(t: T) -> Self {
        HazPtrObj {
            inner: t,
            domain: &SHARED_DOMAIN,
        }
    }
}

impl<T> HazPtrObjBase for HazPtrObj<T> {
    fn deleter(&self) -> &'static dyn Deleter {
        &*DUMMY_DELETER
    }

    fn domain(&self) -> &'static HazPtrDomain {
        self.domain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    struct CountDrops(Arc<AtomicUsize>);
    impl Drop for CountDrops {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub mod deleters {
        fn drop_in_place2(ptr: *mut dyn Drop) {
            // Safe by the contract on HazPtrObject::retire.
            unsafe { std::ptr::drop_in_place(ptr) };
        }
        /// Always safe to use given requirements on HazPtrObject::retire,
        /// but may lead to memory leaks if the pointer type itself needs drop.
        #[allow(non_upper_case_globals)]
        pub static drop_in_place: fn(*mut dyn Drop) = drop_in_place2;
    
        fn drop_box2(ptr: *mut dyn Drop) {
            // Safety: Safe by the safety gurantees of retire and because it's only used when
            // retiring Box objects.
            let _ = unsafe { Box::from_raw(ptr) };
        }
    
        /// # Safety
        ///
        /// Can only be used on values that were originally derived from a Box.
        #[allow(non_upper_case_globals)]
        pub static drop_box: fn(*mut dyn Drop) = drop_box2;
    }

    #[test]
    fn test_write() {
        debug("start");
        let drops_42 = Arc::new(AtomicUsize::new(0));

        let cd = CountDrops(Arc::clone(&drops_42));
        let x = AtomicPtr::new(Box::into_raw(Box::new(
            HazPtrObj::new_with_shared((42, cd))),
        ));
        
        let mut h = HazPtrHolder::default();
        let p2 = h.hazptr() as *const _ as usize;
        let my_x = unsafe { h.load(&x) };
        assert_eq!(SHARED_DOMAIN.activate_count(), 1);

        let drops_9001 = Arc::new(AtomicUsize::new(0));
        let cd2 = CountDrops(Arc::clone(&drops_9001));

        // Change value of x.
        let old = x.swap(
            Box::into_raw(Box::new(HazPtrObj::new_with_shared(
                (9001, cd2)),
            )),
            std::sync::atomic::Ordering::SeqCst,
        );

        let mut h2 = HazPtrHolder::default();

        let p1 = h2.hazptr().next.load(Ordering::SeqCst) as *const _ as usize;
        assert_eq!(p1, p2);

        let my_x2 = unsafe { h2.load(&x) };
        assert_eq!(SHARED_DOMAIN.activate_count(), 2);

        assert_eq!(my_x.inner.0, 42);
        assert_eq!(my_x2.inner.0, 9001);

        // Safety:
        //
        //  1. The pointer came from Box, so is valid.
        //  2. The old value is no longer accessible.
        //  3. The deleter is valid for Box types.
        let old_ref = unsafe {&mut *old};
        assert_eq!(SHARED_DOMAIN.retired_count(), 0);
        // Shall not be reclaimed. Since is held by `h`.
        unsafe { old_ref.retire() };
        assert_eq!(SHARED_DOMAIN.activate_count(), 2);
        assert_eq!(SHARED_DOMAIN.retired_count(), 1);

        assert_eq!(drops_42.load(Ordering::SeqCst), 0);
        assert_eq!(my_x.inner.0, 42);

        debug("must not gc some");
        let n = SHARED_DOMAIN.gc(false);
        assert_eq!(n, 0);

        assert_eq!(drops_42.load(Ordering::SeqCst), 0);
        assert_eq!(my_x.inner.0, 42);

        debug("drop h");
        drop(h);
        assert_eq!(drops_42.load(Ordering::SeqCst), 0);
        // _not_ drop(h2);

        debug("must gc some");
        let n = SHARED_DOMAIN.gc(true);
        assert_eq!(n, 1);

        // assert_eq!(drops_42.load(Ordering::SeqCst), 1);
        // assert_eq!(drops_9001.load(Ordering::SeqCst), 0);

        drop(h2);
        let n = SHARED_DOMAIN.gc(false);
        assert_eq!(n, 0);
        assert_eq!(drops_9001.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_read() {
        debug("start");
        let drops_42 = Arc::new(AtomicUsize::new(0));

        let cd = CountDrops(Arc::clone(&drops_42));
        let x = AtomicPtr::new(Box::into_raw(Box::new(
            HazPtrObj::new_with_shared((42, cd))),
        ));

        // As a reader:
        let mut h = HazPtrHolder::default();

        // Safety:
        //
        //  1. AtomicPtr points to a Box, so is always valid.
        //  2. Writers to AtomicPtr use HazPtrObjBaseect::retire.
        let my_x = unsafe { h.load(&x) };
        // valid:
        assert_eq!(my_x.inner.0, 42);
        h.reset();
        // invalid:
        // let _: i32 = my_x.inner.0;

        let my_x = unsafe { h.load(&x) };
        // valid:
        assert_eq!(my_x.inner.0, 42);
        drop(h);
        // // invalid:
        // let _: i32 = my_x.inner.0;

        let mut h = HazPtrHolder::default();
        let my_x = unsafe { h.load(&x) };

        let mut h_tmp = HazPtrHolder::default();
        let _ = unsafe { h_tmp.load(&x) };
        drop(h_tmp);
    }
}

/// Default Domain and Default Deleters
/// -----------------------------------
/// - Most uses do not need to specify custom domains and custom
///   deleters, and by default use the default domain and default
///   deleters.
///
/// Simple usage example
/// --------------------
///   class Config : public hazptr_obj_base<Config> {
///     /* ... details ... */
///     U get_config(V v);
///   };
///
///   std::atomic<Config*> config_;
///
///   // Called frequently
///   U get_config(V v) {
///     hazptr_holder h = make_hazard_pointer();
///     Config* ptr = h.protect(config_);
///     /* safe to access *ptr as long as it is protected by h */
///     return ptr->get_config(v);
///     /* h dtor resets and releases the owned hazard pointer,
///        *ptr will be no longer protected by this hazard pointer */
///   }
///
///   // called rarely
///   void update_config(Config* new_config) {
///     Config* ptr = config_.exchange(new_config);
///     ptr->retire() // Member function of hazptr_obj_base<Config>
///   }
///
/// Optimized Holders
/// -----------------
/// - The template hazptr_array<M> provides most of the functionality
///   of M hazptr_holder-s but with faster construction/destruction
///   (for M > 1), at the cost of restrictions (on move and swap).
/// - The template hazptr_local<M> provides greater speed even when
///   M=1 (~2 ns vs ~5 ns for construction/destruction) but it is
///   unsafe for the current thread to construct any other holder-type
///   objects (hazptr_holder, hazptr_array and other hazptr_local)
///   while the current instance exists.
/// - In the above example, if Config::get_config() and all of its
///   descendants are guaranteed not to use hazard pointers, then it
///   can be faster (by ~3 ns.) to use
///     hazptr_local<1> h;
///     Config* ptr = h[0].protect(config_);
///  than
///     hazptr_holder h;
///     Config* ptr = h.protect(config_);
///
/// Memory Usage
/// ------------
/// - The size of the metadata for the hazptr library is linear in the
///   number of threads using hazard pointers, assuming a constant
///   number of hazard pointers per thread, which is typical.
/// - The typical number of reclaimable but not yet reclaimed of
///   objects is linear in the number of hazard pointers, which
///   typically is linear in the number of threads using hazard
///   pointers.
///
/// Protecting Linked Structures and Automatic Retirement
/// -----------------------------------------------------
/// Hazard pointers provide link counting API to protect linked
/// structures. It is capable of automatic retirement of objects even
/// when the removal of objects is uncertain. It also supports
/// optimizations when links are known to be immutable. All the link
/// counting features incur no extra overhead for readers.
/// See HazPtrObjBaseLinked.h for more details.
///
/// Alternative Safe Reclamation Methods
/// ------------------------------------
/// - Locking (exclusive or shared):
///   o Pros: simple to reason about.
///   o Cons: serialization, high reader overhead, high contention, deadlock.
///   o When to use: When speed and contention are not critical, and
///     when deadlock avoidance is simple.
/// - Reference counting (atomic shared_ptr):
///   o Pros: automatic reclamation, thread-anonymous, independent of
///     support for thread local data, immune to deadlock.
///   o Cons: high reader (and writer) overhead, high reader (and
///     writer) contention.
///   o When to use: When thread local support is lacking and deadlock
///     can be a problem, or automatic reclamation is needed.
/// - Read-copy-update (RCU):
///   o Pros: simple, fast, scalable.
///   o Cons: sensitive to blocking
///   o When to use: When speed and scalability are important and
///     objects do not need to be protected while blocking.
///
/// Hazard Pointers vs RCU
/// ----------------------
/// - The differences between hazard pointers and RCU boil down to
///   that hazard pointers protect specific objects, whereas RCU
///   sections protect all protectable objects.
/// - Both have comparably low overheads for protection (i.e. reading
///   or traversal) in the order of low nanoseconds.
/// - Both support effectively perfect scalability of object
///   protection by read-only operations (barring other factors).
/// - Both rely on thread local data for performance.
/// - Hazard pointers can protect objects while blocking
///   indefinitely. Hazard pointers only prevent the reclamation of
///   the objects they are protecting.
/// - RCU sections do not allow indefinite blocking, because RCU
///   prevents the reclamation of all protectable objects, which
///   otherwise would lead to deadlock and/or running out of memory.
/// - Hazard pointers can support end-to-end lock-free operations,
///   including updates (provided lock-free allocator), regardless of
///   thread delays and scheduling constraints.
/// - RCU can support wait-free read operations, but reclamation of
///   unbounded objects can be delayed for as long as a single thread
///   is delayed.
/// - The number of unreclaimed objects is bounded when protected by
///   hazard pointers, but is unbounded when protected by RCU.
/// - RCU is simpler to use than hazard pointers (except for the
///   blocking and deadlock issues mentioned above). Hazard pointers
///   need to identify protected objects, whereas RCU does not need to
///   because it protects all protectable objects.
/// - Both can protect linked structures. Hazard pointers needs
///   additional link counting with low or moderate overhead for
///   update operations, and no overhead for readers. RCU protects
///   linked structures automatically, because it implicitly protects
///   all protectable objects.
///
/// Differences from the Standard Proposal
/// --------------------------------------
/// - The latest standard proposal is in wg21.link/p1121. The
///   substance of the proposal was frozen around October 2017, but
///   there were subsequent API changes based on committee feadback.
/// - The main differences are:
///   o This library uses an extra atomic template parameter for
///     testing and debugging.
///   o This library does not support a custom polymorphic allocator
///     (C++17) parameter for the hazptr_domain constructor, until
///     such support becomes widely available.
///   o hazptr_array and hazptr_local are not part of the standard
///     proposal.
///   o Link counting support and protection of linked structures is
///     not part of the current standard proposal.
///   o The standard proposal does not include cohorts and the
///     associated synchronous reclamation capabilities.
struct Dummy {}
