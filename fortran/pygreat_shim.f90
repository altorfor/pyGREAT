! pyGREAT – Fortran shim
! Python supplies background arrays; GREAT does the analysis; results stay in-memory.

module pygreat_shim
  use, intrinsic :: iso_c_binding
  use module_param
  use module_background
  use module_mode_analysis
  use module_eigen

  
  implicit none
  private

  ! ---------------- Saved GREAT state ----------------
  type(parameters_t), save :: S_PARAM
  type(BGData_t),     save :: S_DATA
  logical,            save :: S_DATA_ALLOC = .false.

  ! ---------------- In-shim store of all modes across timesteps ----------------
  type :: StoreMode_t
    integer :: nt = 0
    real(8) :: time = 0.0d0
    integer :: iR = 0
    real(8) :: freq = 0.0d0
    real(8), allocatable :: r(:), nr(:), np_r(:), dp(:), dQ(:), dpsi(:), kcap(:), psic(:)
  end type StoreMode_t

  type(StoreMode_t), allocatable, save :: store(:)
  integer, save :: store_n = 0

  public :: pg_capture_reset
  public :: pg_load_parameters
  public :: pg_set_background
  public :: pg_analyze_current

  public :: pg_modes_count
  public :: pg_mode_shape
  public :: pg_mode_copy

  public :: pg_background_shape
  public :: pg_background_copy

contains

  ! ---------------- Helpers: C-string -> Fortran string ----------------
  pure function cstr_len(cbuf) result(n)
    character(kind=c_char), intent(in) :: cbuf(*)
    integer :: n, i
    integer, parameter :: MAX_SCAN = 4096
    n = 0
    do i = 1, MAX_SCAN
      if (cbuf(i) == c_null_char) exit
      n = n + 1
    end do
  end function cstr_len

  pure function cbuf_to_fstr(cbuf) result(s)
    character(kind=c_char), intent(in) :: cbuf(*)
    character(len=cstr_len(cbuf)) :: s
    integer :: i
    do i = 1, len(s)
      s(i:i) = transfer(cbuf(i), 'a')
    end do
  end function cbuf_to_fstr

  ! ---------------- Store management ----------------
  subroutine store_reset()
    integer :: k
    if (allocated(store)) then
      do k = 1, size(store)
        if (allocated(store(k)%r))    deallocate(store(k)%r)
        if (allocated(store(k)%nr))   deallocate(store(k)%nr)
        if (allocated(store(k)%np_r)) deallocate(store(k)%np_r)
        if (allocated(store(k)%dp))   deallocate(store(k)%dp)
        if (allocated(store(k)%dQ))   deallocate(store(k)%dQ)
        if (allocated(store(k)%dpsi)) deallocate(store(k)%dpsi)
        if (allocated(store(k)%kcap))    deallocate(store(k)%kcap)
        if (allocated(store(k)%psic))  deallocate(store(k)%psic)
      end do
      deallocate(store)
    end if
    store_n = 0
  end subroutine store_reset

  subroutine store_append(nt, time, iR, freq, r, nr, np_r, dp, dQ, dpsi, K, Psi)
    integer, intent(in) :: nt, iR
    real(8), intent(in) :: time, freq
    real(8), intent(in) :: r(iR), nr(iR), np_r(iR), dp(iR), dQ(iR), dpsi(iR), K(iR), Psi(iR)
    type(StoreMode_t), allocatable :: tmp(:)
    integer :: n

    n = store_n + 1
    if (.not.allocated(store)) then
      allocate(store(1))
    else
      allocate(tmp(n))
      if (store_n > 0) tmp(1:store_n) = store(1:store_n)
      call move_alloc(tmp, store)
    end if

    store(n)%nt   = nt
    store(n)%time = time
    store(n)%iR   = iR
    store(n)%freq = freq

    allocate(store(n)%r(iR), store(n)%nr(iR), store(n)%np_r(iR))
    allocate(store(n)%dp(iR), store(n)%dQ(iR), store(n)%dpsi(iR))
    allocate(store(n)%kcap(iR), store(n)%psic(iR))

    store(n)%r    = r
    store(n)%nr   = nr
    store(n)%np_r = np_r
    store(n)%dp   = dp
    store(n)%dQ   = dQ
    store(n)%dpsi = dpsi
    store(n)%kcap    = K
    store(n)%psic  = Psi

    store_n = n
  end subroutine store_append

  ! ---------------- C ABI: resets accumulated modes & GREAT capture ----------------
  subroutine pg_capture_reset() bind(C, name="pg_capture_reset")
    call store_reset()
    call eigen_capture_reset()
  end subroutine pg_capture_reset

  ! ---------------- C ABI: load GREAT parameters from file ----------------
  subroutine pg_load_parameters(c_parfile, ierr_c) bind(C, name="pg_load_parameters")
    character(kind=c_char), intent(in) :: c_parfile(*)
    integer(c_int),         intent(out):: ierr_c
    character(len=:), allocatable :: parfile
    integer :: ierr

    parfile = cbuf_to_fstr(c_parfile)
    ierr    = 0
    call Read_parameters(S_PARAM, parfile, ierr)
    ! Force GREAT to avoid writing eigenfiles; Python will handle outputs
    S_PARAM%output_eigen = .false.
    ierr_c = ierr
  end subroutine pg_load_parameters

  ! ---------------- C ABI: set one time-slice background ----------------
  subroutine pg_set_background(nt_c, time_c, m_c, r, rho, eps, p, cs2, phi, alpha, v1, ierr_c) bind(C, name="pg_set_background")
    integer(c_int),  value      :: nt_c, m_c
    real(c_double),  value      :: time_c
    real(c_double),  intent(in) :: r(*), rho(*), eps(*), p(*), cs2(*), phi(*), alpha(*), v1(*)
    integer(c_int),  intent(out):: ierr_c

    integer :: ierr, m, i
    m = m_c
    ierr = 0

    ! (Re)allocate background
    if (S_DATA_ALLOC) call Free_background_data(S_DATA, ierr)
    call Init_background_data(S_DATA, m, ierr)
    if (ierr /= 0) then
      ierr_c = ierr; return
    end if
    S_DATA_ALLOC = .true.

    ! Tag time / index
    S_DATA%nt   = nt_c
    S_DATA%time = time_c
    S_DATA%num_t = 1
    S_DATA%iR   = 0

    ! Copy arrays (1..m)
    do i = 1, m
      S_DATA%r(i)               = r(i)
      S_DATA%rho(i)             = rho(i)
      S_DATA%eps(i)             = eps(i)
      S_DATA%p(i)               = p(i)
      S_DATA%c_sound_squared(i) = cs2(i)
      S_DATA%phi(i)             = phi(i)
      S_DATA%alpha(i)           = alpha(i)
      S_DATA%v_1(i)             = v1(i)
    end do

    ierr_c = 0
  end subroutine pg_set_background

  ! ---------------- C ABI: analyze current background ----------------
  subroutine pg_analyze_current(ierr_c) bind(C, name="pg_analyze_current")
    integer(c_int), intent(out) :: ierr_c
    integer :: ierr, nm, k, iR, ier2
    real(8) :: freq
    real(8), allocatable :: r(:), nr(:), np_r(:), dp(:), dQ(:), dpsi(:), kcap(:), psic(:)

    ierr = 0
    if (.not.S_DATA_ALLOC) then
      ierr_c = 900; return
    end if

    ! Detect shock / outer boundary
    call Compute_shock_location_bg(S_DATA, S_PARAM, ierr)
    if (ierr /= 0 .or. S_DATA%iR == 0) then
      ierr_c = 901; return
    end if

    ! Fresh GREAT per-nt capture
    call eigen_capture_reset()

    ! Run the analysis; module_mode_analysis will call eigen_capture_append per mode
    call Perform_mode_analysis(S_DATA, S_PARAM, ierr)
    if (ierr /= 0) then
      ierr_c = ierr; return
    end if

    ! Pull modes from GREAT capture and append to the shim store tagged with nt/time
    call eigen_modes_count(nm)
    do k = 1, nm
      call eigen_mode_shape(k, iR)
      if (iR <= 0) cycle
        allocate(r(iR), nr(iR), np_r(iR), dp(iR), dQ(iR), dpsi(iR), kcap(iR), psic(iR), stat=ier2)
      if (ier2 /= 0) cycle

      call eigen_mode_copy(k, r, nr, np_r, dp, dQ, dpsi, kcap, psic, freq, ier2)
      if (ier2 == 0) then
        call store_append(S_DATA%nt, S_DATA%time, iR, freq, r, nr, np_r, dp, dQ, dpsi, kcap, psic)
      end if

      deallocate(r, nr, np_r, dp, dQ, dpsi, kcap, psic, stat=ier2)
    end do

    ierr_c = 0
  end subroutine pg_analyze_current

  ! ---------------- C ABI: modes getters ----------------
  subroutine pg_modes_count(n_c) bind(C, name="pg_modes_count")
    integer(c_int), intent(out) :: n_c
    n_c = store_n
  end subroutine pg_modes_count

  subroutine pg_mode_shape(k_c, iR_c) bind(C, name="pg_mode_shape")
    integer(c_int), value :: k_c
    integer(c_int), intent(out) :: iR_c
    if (k_c < 1 .or. k_c > store_n) then
      iR_c = 0
    else
      iR_c = store(k_c)%iR
    end if
  end subroutine pg_mode_shape

  subroutine pg_mode_copy(k_c, r, nr, np_over_r, dp, dQ, dpsi, K_cap, Psi_cap, freq_c, nt_c, time_c, ierr_c) bind(C, name="pg_mode_copy")
    integer(c_int), value :: k_c
    real(c_double), intent(out) :: r(*), nr(*), np_over_r(*), dp(*), dQ(*), dpsi(*), K_cap(*), Psi_cap(*)
    real(c_double), intent(out) :: freq_c, time_c
    integer(c_int),  intent(out):: nt_c, ierr_c
    integer :: i, iR

    if (k_c < 1 .or. k_c > store_n) then
      ierr_c = 1; nt_c = 0; freq_c = 0.0d0; time_c = 0.0d0
      return
    end if

    iR = store(k_c)%iR
    do i = 1, iR
      r(i)         = store(k_c)%r(i)
      nr(i)        = store(k_c)%nr(i)
      np_over_r(i) = store(k_c)%np_r(i)
      dp(i)        = store(k_c)%dp(i)
      dQ(i)        = store(k_c)%dQ(i)
      dpsi(i)      = store(k_c)%dpsi(i)
      K_cap(i)     = store(k_c)%kcap(i)
      Psi_cap(i)   = store(k_c)%psic(i)
    end do

    freq_c = store(k_c)%freq
    nt_c   = store(k_c)%nt
    time_c = store(k_c)%time
    ierr_c = 0
  end subroutine pg_mode_copy

  ! ---------------- C ABI: background copy (last analyzed nt) ----------------
  subroutine pg_background_shape(iR_c) bind(C, name="pg_background_shape")
    integer(c_int), intent(out) :: iR_c
    if (.not.S_DATA_ALLOC) then
      iR_c = 0
    else
      iR_c = S_DATA%iR
    end if
  end subroutine pg_background_shape

  subroutine pg_background_copy(r, rho, eps, p, cs2, phi, alpha, h, q, e, gamma1, ggrav, n2, lamb2, Bstar, Qcap, inv_cs2, mass_v, nt_c, time_c, ierr_c) bind(C, name="pg_background_copy")
    real(c_double), intent(out) :: r(*), rho(*), eps(*), p(*), cs2(*), phi(*), alpha(*)
    real(c_double), intent(out) :: h(*), q(*), e(*), gamma1(*), ggrav(*), n2(*), lamb2(*), Bstar(*), Qcap(*), inv_cs2(*), mass_v(*)
    integer(c_int),  intent(out) :: nt_c, ierr_c
    real(c_double),  intent(out) :: time_c

    integer :: i, iR

    if (.not.S_DATA_ALLOC .or. S_DATA%iR <= 0) then
      ierr_c = 1; nt_c = 0; time_c = 0.0d0; return
    end if

    iR = S_DATA%iR
    do i = 1, iR
      r(i)       = S_DATA%r(i)
      rho(i)     = S_DATA%rho(i)
      eps(i)     = S_DATA%eps(i)
      p(i)       = S_DATA%p(i)
      cs2(i)     = S_DATA%c_sound_squared(i)
      phi(i)     = S_DATA%phi(i)
      alpha(i)   = S_DATA%alpha(i)
      h(i)       = S_DATA%h(i)
      q(i)       = S_DATA%q(i)
      e(i)       = S_DATA%e(i)
      gamma1(i)  = S_DATA%gamma_one(i)
      ggrav(i)   = S_DATA%ggrav(i)
      n2(i)      = S_DATA%n2(i)
      lamb2(i)   = S_DATA%lamb2(i)
      Bstar(i)   = S_DATA%Bstar(i)
      Qcap(i)    = S_DATA%Qcap(i)
      inv_cs2(i) = S_DATA%inv_cs2(i)
      mass_v(i)  = S_DATA%mass_v(i)
    end do

    nt_c   = S_DATA%nt
    time_c = S_DATA%time
    ierr_c = 0
  end subroutine pg_background_copy

end module pygreat_shim
