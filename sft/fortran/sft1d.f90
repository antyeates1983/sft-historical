!****************************************************************
PROGRAM sft1d
!----------------------------------------------------------------
! 1D SFT - solves for longitude-averaged B_r
! [note: inputs and most outputs are same as 2D version]
!----------------------------------------------------------------

IMPLICIT NONE

    INTEGER, PARAMETER :: d=KIND(0.0d0)  ! double precision
    REAL(d), PARAMETER :: PI=4d0*ATAN(1d0)
    REAL(d), PARAMETER :: CFL=0.2d0
    INTEGER, PARAMETER :: outfreq_hrs = 24
    
    CHARACTER(LEN=200) :: datapath, ns_arg, np_arg, hr_end_arg, procid
    CHARACTER(LEN=200) :: hale_factor_arg
    CHARACTER(LEN=21), ALLOCATABLE :: filename_regs(:)
    LOGICAL :: quenching_flag
    INTEGER :: i, io, ns, np, hr_end
    INTEGER :: hr, hr_evolve, hr_evolve_reg, hr_evolve_out, hr_last_out
    INTEGER :: k_out, k_regs, nregs, nouts, nsteps
    INTEGER, ALLOCATABLE :: hr_regs(:), hr_out(:)
    REAL(d) :: hale_factor
    REAL(d) :: ds, dp, np_d
    REAL(d) :: t_evolve, dtmax, dt_mf, dt_eta, dt, dtds
    REAL(d) :: omA, omB, omC, v0, p0, eta0, tau, bq
    REAL(d), ALLOCATABLE :: br(:), br0(:,:), br2(:,:)
    REAL(d), ALLOCATABLE :: sc(:), sg(:), s2c(:), sfact(:)
    REAL(d), ALLOCATABLE :: fs(:), vs(:)
    REAL(d), ALLOCATABLE :: vs_l(:), vs_r(:)
    REAL(d), ALLOCATABLE :: dip_out(:), uflux_out(:), bfly_out(:,:)

    ! Read command line parameters:
    CALL GET_COMMAND_ARGUMENT(1, datapath)
    CALL GET_COMMAND_ARGUMENT(2, procid)
    CALL GET_COMMAND_ARGUMENT(3, ns_arg)   ! needed still for input data
    CALL GET_COMMAND_ARGUMENT(4, np_arg)
    CALL GET_COMMAND_ARGUMENT(5, hr_end_arg)
    READ(ns_arg,*) ns
    READ(np_arg,*) np
    READ(hr_end_arg,*) hr_end

    IF (COMMAND_ARGUMENT_COUNT() > 5) THEN
        CALL GET_COMMAND_ARGUMENT(6, hale_factor_arg)
        READ(hale_factor_arg, *) hale_factor
    END IF

    ! Initialization:
    np_d = REAL(np)
    ! - grid spacing:
    ds = 2d0/REAL(ns)
    dp = 2d0*PI/REAL(np)
    ! - declare arrays:
    ALLOCATE(br(1:ns), br0(1:ns,1:np), br2(1:np,1:ns))
    ALLOCATE(fs(0:ns))
    ALLOCATE(sg(1:ns-1), sc(1:ns), sfact(1:ns-1))
    ALLOCATE(vs(1:ns-1), vs_l(1:ns-1), vs_r(1:ns-1))
    ! - read in initial br, and flow parameters:
    OPEN(1, file=TRIM(datapath)//'br0'//TRIM(procid)//'.unf', form='unformatted')
    READ(1) br0
    READ(1) omA  ! not used in 1D
    READ(1) omB  ! not used in 1D
    READ(1) omC  ! not used in 1D
    READ(1) v0
    READ(1) p0
    READ(1) eta0
    READ(1) bq  ! not used in 1D
    READ(1) tau
    CLOSE(1)
    br = SUM(br0, DIM=2)/np_d

    ! - initialise velocity arrays and coordinate factors:
    sc = (/ (-1d0 + (REAL(i) - 0.5d0)*ds, i = 1, ns)/)
    sg = (/ (-1d0 + REAL(i)*ds, i = 1, ns-1)/)
    vs(1:ns-1) = v0*(1d0+p0)**(0.5d0*(p0+1d0))/p0**(0.5d0*p0) * sg * (SQRT(1d0 - sg**2))**p0
    sfact(1:ns-1) = 1.0_d - sg**2
    vs_l = 0.5_d*(1.0_d + SIGN(1.0_d,vs))*vs
    vs_r = 0.5_d*(1.0_d - SIGN(1.0_d,vs))*vs
    sfact = sfact*eta0/ds
    ! - maximum timestep:
    dt_mf = MINVAL(ds/ABS(vs))
    dt_eta = ds*dp**2/eta0
    dtmax = CFL * MINVAL((/dt_eta, dt_mf/))
    ! - include factor sin(theta) in vs:
    vs_l(1:ns-1) = vs_l(1:ns-1) * SQRT(1 - sg**2)
    vs_r(1:ns-1) = vs_r(1:ns-1) * SQRT(1 - sg**2)
    ! - read in emergence times and corresponding filenames:
    OPEN(15, FILE=TRIM(datapath)//'timefile.txt', STATUS='OLD', FORM='FORMATTED', ACTION='READ')
    ! -- first count number of regions:
    nregs = 0
    DO
        READ(15, *, IOSTAT=io)
        IF (io /= 0) EXIT
        nregs = nregs + 1
    END DO
    REWIND(15)
    ! -- read lists of emergence hours and filenames:
    ALLOCATE(hr_regs(1:nregs+1), filename_regs(1:nregs+1))
    DO i=1, nregs
        READ(15,*) hr_regs(i), filename_regs(i)
    END DO
    CLOSE(15)
    hr_regs(nregs+1) = hr_end  ! for convenience finishing the simulation
    ! - initialise output arrays:
    nouts = hr_end/outfreq_hrs
    ALLOCATE(hr_out(1:nouts), uflux_out(1:nouts), dip_out(1:nouts), bfly_out(1:nouts,1:ns))
    hr_out(1) = 0
    dip_out(1) = 1.5d0 * SUM(br * sc * ds)
    uflux_out(1) = SUM(ABS(br) * ds)
    bfly_out(1,1:ns) = br

    ! Main loop:
    k_out = 2
    k_regs = 1
    hr_last_out = 0
    hr = 0
    DO WHILE (hr < hr_end)
!        PRINT*, 'hour', hr, MAXVAL(ABS(br))
        
        ! Insert any regions:
        DO WHILE ((hr == hr_regs(k_regs)).AND.(k_regs <= nregs))
            IF (COMMAND_ARGUMENT_COUNT() > 5) THEN
                CALL add_region(br2, TRIM(datapath)//'regions/'//filename_regs(k_regs), hale_factor)
            ELSE
                CALL add_region(br2, TRIM(datapath)//'regions/'//filename_regs(k_regs), 0d0)
            END IF
            br = br + SUM(br2, DIM=1)/np_d
            k_regs = k_regs + 1
        END DO

        ! Output if required:
        IF (hr == hr_last_out + outfreq_hrs) THEN
            hr_last_out = hr
            hr_out(k_out) = hr
            uflux_out(k_out) = SUM(ABS(br) * ds)
            dip_out(k_out) = 1.5d0 * SUM(br * sc * ds)
            bfly_out(k_out,1:ns) = br
            k_out = k_out + 1
        END IF
        
        ! How long to evolve for:
        hr_evolve_out = hr_last_out + outfreq_hrs - hr
        hr_evolve_reg = hr_regs(k_regs) - hr
        hr_evolve = min(hr_evolve_out, hr_evolve_reg)
        
        ! Set timestep to divide evenly:
        t_evolve = REAL(hr_evolve * 3600)   ! in secs
        nsteps = INT(t_evolve/dtmax)
        IF (nsteps == 0) nsteps = 1
        dt = t_evolve/REAL(nsteps)
        dtds = dt/ds
        
        ! Do the evolution:
        CALL evolve_expl(nsteps)
    
         hr = hr + hr_evolve

    END DO

    ! Output results to file:
    OPEN(1, file=TRIM(datapath)//'outs'//TRIM(procid)//'.unf', form='unformatted')
    WRITE(1) nouts
    WRITE(1) hr_out
    WRITE(1) uflux_out  ! for backward-compatibility [not equivalent to 2d]
    WRITE(1) dip_out
    WRITE(1) bfly_out
    WRITE(1) bfly_out  ! for backward-compatibility [this is unsigned bfly in 2d]
    CLOSE(1)

CONTAINS

!==========================================================
SUBROUTINE add_region(br2, rfile, hale_factor)
    ! Read 2D 'br' array from binary file.
    REAL(d), DIMENSION(:,:), INTENT(INOUT) :: br2
    CHARACTER*(*), INTENT(IN) :: rfile
    REAL(d), INTENT(IN) :: hale_factor
    REAL(d), DIMENSION(:,:), ALLOCATABLE :: tmp
    REAL(d) :: tmp1
    INTEGER(KIND=2) :: tmp2
    INTEGER(KIND=1) :: complete

    OPEN(1, file=TRIM(rfile), form='unformatted')
    READ(1) br2

    IF (hale_factor > 1d-10) THEN
        ALLOCATE(tmp, MOLD=br2)
        READ(1) tmp
        READ(1) tmp1
        READ(1) tmp2
        READ(1) complete
        IF (complete == 0) br2 = br2 / 1.3d0 * hale_factor
    END IF

    CLOSE(1)

END SUBROUTINE add_region

SUBROUTINE evolve_expl(nsteps)
    ! Evolve br by flux transport evolution for nsteps time steps.
    ! Use simple explicit finite-volume method with upwinding for advection term.
    ! Boundary conditions are zero flux at the poles and periodic in phi.
    INTEGER, INTENT(IN) :: nsteps
    INTEGER :: k

    fs = 0d0

    DO k = 1, nsteps
        ! Evaluate finite-volume fluxes [at interior ribs]:
        ! - diffusion:
        fs(1:ns-1) = sfact*(br(2:ns) - br(1:ns-1))
        ! - meridional flow (by upwinding):
        fs(1:ns-1) = fs(1:ns-1) - vs_l(1:ns-1)*br(1:ns-1) - vs_r(1:ns-1)*br(2:ns)
        ! Update:
        br(1:ns) = br(1:ns) + dtds*(fs(1:ns) - fs(0:ns-1)) 
        
        ! Add exponential decay term:
        IF (tau > 0) br(1:ns) = br(1:ns) - dt/tau * br(1:ns)
    END DO

END SUBROUTINE evolve_expl

END PROGRAM sft1d
