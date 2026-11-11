//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

namespace {
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real GetNu(Real R, Real T);

void GetCylCoord(Coordinates *pco, Real &rad, Real &phi, Real &z, int i, int j, int k);
Real Logx1(Real x, RegionSize rs);

// problem parameters which are useful to make global to this file
Real rad_soft, gmp, rad_planet, phi_planet_0, z_planet,
t0_planet, vk_planet, omega_planet, inv_omega_planet, cs_planet,
gm0, r0, rho0, dslope, p0_over_r0, pslope, nu_alpha,
t_planet_growth, dfloor, Omega0,
x1min, x1max, damping_rate, Hill_radius,
radius_inner_damping, radius_outer_damping, inner_ratio_region,
outer_ratio_region, inner_width_damping, outer_width_damping;

Real mdot_in = 0.0;
bool fix_outer_mass_flux = false;

bool Damping_Flag, Isothermal_Flag;
int PlanetaryGravityOrder;

// User-defined Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_s);
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_s);
void MyVisc(HydroDiffusion *phdif, MeshBlock *pmb,
  const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,
  int is, int ie, int js, int je, int ks, int ke);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0                      = pin->GetOrAddReal("problem", "GM", 0.0);
  r0                       = pin->GetOrAddReal("problem", "r0", 1.0);
  Damping_Flag             = pin->GetBoolean("problem", "Damping_Flag");
  Isothermal_Flag          = pin->GetBoolean("problem", "Isothermal_Flag");

  // Get parameters for initial density and velocity
  rho0      = pin->GetReal("problem", "rho0");
  dslope    = pin->GetOrAddReal("problem", "dslope", -1.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
    pslope     = pin->GetOrAddReal("problem", "pslope",     -0.5);
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor    = pin->GetOrAddReal("hydro", "dfloor", (1024*(float_min)));
  nu_alpha  = pin->GetOrAddReal("problem", "nu_alpha", 0.0);
  Omega0    = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);

  // Prescribed total inward mass accretion rate
  mdot_in = pin->GetOrAddReal("problem", "mdot_in", 0.0);

  fix_outer_mass_flux = pin->GetOrAddBoolean("problem", "fix_outer_mass_flux", false);

if (mdot_in < 0.0) {
  throw std::runtime_error("mdot_in must be non-negative");
}

if (fix_outer_mass_flux &&
    mesh_bcs[BoundaryFace::outer_x1] != GetBoundaryFlag("user")) {
  throw std::runtime_error(
      "fix_outer_mass_flux requires ox1_bc = user");
}

  // The parameters of one planet
  rad_planet   = pin->GetOrAddReal("problem",  "rad_planet", 1.0);            // radial position of the planet
  phi_planet_0 = pin->GetOrAddReal("problem",  "phi_planet", 0.0);            // azimuthal position of the planet
  z_planet     = pin->GetOrAddReal("problem",  "z_planet",   0.0);            // vertical position of the planet
  t0_planet    = (pin->GetOrAddReal("problem", "t0_planet",  0.0))*TWO_PI*r0; // time to put in the planet
  gmp          = pin->GetOrAddReal("problem",  "GMp",        0.0);            // GM of the planet

  vk_planet        = std::sqrt(gm0/rad_planet);
  omega_planet     = vk_planet/rad_planet;
  inv_omega_planet = 1.0/omega_planet;

  PlanetaryGravityOrder = pin->GetOrAddInteger("problem", "PlanetaryGravityOrder", 2);
  if ((PlanetaryGravityOrder != 2) && (PlanetaryGravityOrder != 4) && (PlanetaryGravityOrder != 6))
    PlanetaryGravityOrder = 2;

  if (NON_BAROTROPIC_EOS)
    cs_planet = std::sqrt(p0_over_r0*std::pow(rad_planet/r0, pslope));
  else
    cs_planet = std::sqrt(p0_over_r0);

  t_planet_growth  = pin->GetOrAddReal("problem", "t_planet_growth", 0.0); // time to grow the planet

  Hill_radius = (std::pow(gmp/gm0*ONE_3RD, ONE_3RD)*rad_planet);

  rad_soft  = pin->GetOrAddReal("problem", "rs", 0.6); // softening length of the gravitational potential of planets
  rad_soft *= Hill_radius;

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  // x1 coordinate
  if (pin->GetReal("mesh","x1rat") < 0.0){
    EnrollUserMeshGenerator(X1DIR, Logx1);
  }

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*std::pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*std::pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }


  // Enroll planetary gravity, damping zone, local isothermal and beta cooling
  EnrollUserExplicitSourceFunction(MySource);

  EnrollViscosityCoefficient(MyVisc);

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  AllocateUserOutputVariables(6);
  SetUserOutputVariableName(0, "flx_x1_in");
  SetUserOutputVariableName(1, "flx_x1_out");

  SetUserOutputVariableName(2, "mom_r_flux_in");
  SetUserOutputVariableName(3, "mom_r_flux_out");

  SetUserOutputVariableName(4, "mom_phi_flux_in");
  SetUserOutputVariableName(5, "mom_phi_flux_out");

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);

  Real igm1 = 1.0/(peos->GetGamma() - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

    // Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
        Real cs_square = PoverRho(rad, phi, z);
        Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
        Real vel_K     = vK(porb, x1, x2, x3);

        // compute initial conditions in cylindrical coordinates
        Real den_gas   = DenProfileCyl_gas(rad, phi, z);
        Real vis_vel_r = -1.5*(nu_alpha*cs_square/rad/omega_dyn);

        Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
        if (porb->orbital_advection_defined)
          vel_gas_phi -= vK(porb, x1, x2, x3);

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);
        

        gas_dens = den_gas;
        gas_mom1 = den_gas*vis_vel_r;

        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          gas_mom2 = den_gas*vel_gas_phi;
          gas_mom3 = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          gas_mom2 = 0.0;
          gas_mom3 = den_gas*vel_gas_phi;
        }

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i)  = cs_square*phydro->u(IDN, k, j, i)*igm1;
          phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                        + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
        }

      }
    }
  }

  // Initialize prescribed outer-boundary mass flux
  phydro->use_inflx_x1 = fix_outer_mass_flux;

  if (fix_outer_mass_flux) {
    const Real r_out = x1max;

    const Real flux_density =
        -mdot_in / (TWO_PI * r_out);

    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        phydro->inflx_x1(k,j,0) = flux_density;
      }
    }
  }

  return;
}


namespace {
//----------------------------------------------------------------------------------------
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_s) {

  if ((gmp > 0.0) && (time >= t0_planet))
    PlanetaryGravity(pmb, time, dt, prim, prim_s, bcc, cons, cons_s);

  return;
}

void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_s) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real phi_planet_move    = omega_planet*time + phi_planet_0;
  Real planet_gm;
  int nc1 = pmb->ncells1;

  phi_planet_move -= Omega0*time;

  if (time >= t0_planet) {
    AthenaArray<Real> distance_square;
    AthenaArray<Real> x_dis, y_dis, z_dis, rad_dis, phi_dis, theta_dis;
    AthenaArray<Real> rad_arr, phi_arr, z_arr;
    AthenaArray<Real> acc_r, acc_phi, acc_z, indirect_acc_r, indirect_acc_phi;

    distance_square.NewAthenaArray(nc1);

    x_dis.NewAthenaArray(nc1);
    y_dis.NewAthenaArray(nc1);
    z_dis.NewAthenaArray(nc1);

    rad_dis.NewAthenaArray(nc1);
    phi_dis.NewAthenaArray(nc1);
    theta_dis.NewAthenaArray(nc1);

    rad_arr.NewAthenaArray(nc1);
    phi_arr.NewAthenaArray(nc1);
    z_arr.NewAthenaArray(nc1);

    acc_r.NewAthenaArray(nc1);
    acc_phi.NewAthenaArray(nc1);
    acc_z.NewAthenaArray(nc1);

    indirect_acc_r.NewAthenaArray(nc1);
    indirect_acc_phi.NewAthenaArray(nc1);

    planet_gm = gmp*std::sin(0.5*PI*std::min((time-t0_planet)/t_planet_growth, 1.0));

    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

          x_dis(i) = rad_arr(i)*std::cos(phi_arr(i)) - rad_planet*std::cos(phi_planet_move);
          y_dis(i) = rad_arr(i)*std::sin(phi_arr(i)) - rad_planet*std::sin(phi_planet_move);
          z_dis(i) = z_arr(i) - z_planet;

          rad_dis(i) = rad_arr(i) - rad_planet*std::cos(phi_arr(i) - phi_planet_move);
          phi_dis(i) = rad_planet*std::sin(phi_arr(i) - phi_planet_move);

          distance_square(i) = SQR(x_dis(i)) + SQR(y_dis(i)) + SQR(z_dis(i));

          indirect_acc_r(i)   =  planet_gm*std::cos(phi_arr(i) - phi_planet_move)/SQR(rad_planet);
          indirect_acc_phi(i) = -planet_gm*std::sin(phi_arr(i) - phi_planet_move)/SQR(rad_planet);

            //second order gravity
          if (PlanetaryGravityOrder == 2) {
            Real sec_g = planet_gm/std::pow(distance_square(i)+SQR(rad_soft), 1.5);
            acc_r(i)   = sec_g*rad_dis(i) + indirect_acc_r(i);   // radial acceleration
            acc_phi(i) = sec_g*phi_dis(i) + indirect_acc_phi(i); // asimuthal acceleration
            acc_z(i)   = sec_g*z_dis(i);                         // vertical acceleartion
          }

          //fourth order gravity
          if (PlanetaryGravityOrder == 4) {
            Real forth_g = planet_gm*(5.0*SQR(rad_soft)+2.0*distance_square(i))/
                                    (2.0*std::pow(SQR(rad_soft)+distance_square(i), 2.5));
            acc_r(i)   = forth_g*rad_dis(i) + indirect_acc_r(i);   // radial acceleration
            acc_phi(i) = forth_g*phi_dis(i) + indirect_acc_phi(i); // asimuthal acceleration
            acc_z(i)   = forth_g*z_dis(i);                         // vertical acceleartion
          }

          //sixth order gravity
          if (PlanetaryGravityOrder == 6) {
            Real sixth_g = planet_gm*(35.0*SQR(SQR(rad_soft))+28.0*SQR(rad_soft)*distance_square(i)+
                                    8.0*distance_square(i))/(8.0*std::pow(SQR(rad_soft)+distance_square(i), 3.5));
            acc_r(i)   = sixth_g*rad_dis(i) + indirect_acc_r(i);   // radial acceleration
            acc_phi(i) = sixth_g*phi_dis(i) + indirect_acc_phi(i); // asimuthal acceleration
            acc_z(i)   = sixth_g*z_dis(i);                         // vertical acceleartion
          }

          const Real &gas_rho  = prim(IDN, k, j, i);
          const Real &gas_vel1 = prim(IM1, k, j, i);
          const Real &gas_vel2 = prim(IM2, k, j, i);
          const Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);

          Real delta_mom1 = -dt*gas_rho*acc_r(i);
          Real delta_mom2 = -dt*gas_rho*acc_phi(i);
          Real delta_mom3 = -dt*gas_rho*acc_z(i);

          gas_mom1 += delta_mom1;
          gas_mom2 += delta_mom2;
          gas_mom3 += delta_mom3;

          if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
            Real &gas_erg  = cons(IEN, k, j, i);
            gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
          }
        }

      }
    }
  }
  return;
}

Real Logx1(Real x, RegionSize rs)
{
  // generate log spacing
  Real w= std::log10(rs.x1min) + x*(std::log10(rs.x1max/rs.x1min));
  return std::pow(10,w);
}

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad = pco->x1v(i);
    phi = pco->x2v(j);
    z   = pco->x3v(k);
  }

  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi = pco->x3v(i);
    z   = pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den, dfloor);
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates
Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //if (sigma_active < TINY_NUMBER)
    //vr_ghost = vr_active >= 0.0 ? ((sigma_active+TINY_NUMBER)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //else
  //vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}

Real GetNu(Real R, Real cs2){
  // alpha viscosity
  Real Omega = sqrt(gm0/(R*R*R));
  Real nu_disk = nu_alpha * cs2 / Omega;
  
  return nu_disk;
}

void MyVisc(HydroDiffusion *phdif, MeshBlock *pmb,
  const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,
  int is, int ie, int js, int je, int ks, int ke) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        Real cs2 = PoverRho(rad, phi, z);
        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = GetNu(rad, cs2);
      }
    }
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad_ghost, phi_ghost, z_ghost;
        GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, il-i, j, k);

        Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
        Real pre_diff  = (pslope + dslope)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
        Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
        Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
        Real &gas_vel3_ghost = prim(IM3, k, j, il-i);

        Real vis_vel_r   = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        if (pmb->porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;

          /*
          std::cout << "vr_ghost = " << vis_vel_r <<
                    "cs_2 = " << cs_square <<
                    "nu = " << nu_alpha * cs_square /omega_dyn <<
                     pmb->phydro->hdif.nu(HydroDiffusion::DiffProcess::iso, k, j, il-i) << std::endl;*/

        // gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        // gas_rho_ghost  = prim(IDN, k, j, il);

        Real p_over_r = p0_over_r0;
        if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real denmid = 1*std::pow(rad_ghost/r0,-1.5-pslope);
        Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad_ghost)+SQR(z_ghost))-1./rad_ghost));
        Real den = dentem;

        gas_rho_ghost  = std::max(den,dfloor);
        gas_vel1_ghost = vis_vel_r;
        gas_vel2_ghost = prim(IM2, k ,j, il);
        gas_vel3_ghost = 0.0;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pres_ghost = prim(IEN, k, j, il-i);
          gas_pres_ghost       = cs_square*gas_rho_ghost;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad_ghost, phi_ghost, z_ghost;
        GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, iu+i, j, k);

        Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vel_K     = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
        Real pre_diff  = (pslope + dslope)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
        Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
        Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
        Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);

        Real vis_vel_r   = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        if (pmb->porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;

        // gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        // gas_rho_ghost = prim(IDN, k, j, iu);

        Real p_over_r = p0_over_r0;
        if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real denmid = 1*std::pow(rad_ghost/r0,-1.5-pslope);
        // Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad_ghost)+SQR(z_ghost))-1./rad_ghost));
        Real den = denmid;

        gas_rho_ghost = prim(IDN, k, j, iu);
        gas_vel1_ghost = prim(IM1, k ,j, iu);
        gas_vel2_ghost = prim(IM2, k ,j, iu);
        gas_vel3_ghost = 0.0;
        if (NON_BAROTROPIC_EOS) {
          Real &gas_pres_ghost = prim(IEN, k, j, iu+i);
          gas_pres_ghost       = cs_square*gas_rho_ghost;
        }
      }
    }
  }
}

void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        Real &gas_pres = prim(IPR, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real inv_gas_dens = 1.0/gas_dens;
        gas_pres = PoverRho(rad, phi, z)*gas_dens;
        gas_erg  = gas_pres*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
      }
    }
  }
  return;
}


void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  Real v2_phi, v2_theta, v3_phi, v3_theta;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v2_phi = 1.0; v2_theta = 0.0;
    v3_phi = 0.0; v3_theta = 1.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v2_phi = 0.0; v2_theta = 1.0;
    v3_phi = 1.0; v3_theta = 0.0;
  }

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) <= radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_r      = -1.5*(nu_alpha*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_theta  = 0.0;
          Real vel_gas_phi    = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi        -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_r;
          Real gas_vel2_0 = v2_phi*vel_gas_phi + v2_theta*vel_gas_theta;
          Real gas_vel3_0 = v3_phi*vel_gas_phi + v3_theta*vel_gas_theta;
          Real gas_pre_0  = cs_square_0*gas_rho_0;

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);
          Real &gas_pre  = prim(IPR, k, j, i);

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_0  - gas_pre )*R_func(i)*inv_damping_tau(i)*dt;

          // for damping zone, one degree of freedom needs to be left for the hydro system to relax itself.
          // This is purely based on experience.
          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            delta_gas_vel2 = 0.0;
          } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            delta_gas_vel3 = 0.0;
          }

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;
          gas_pre  += delta_gas_pre;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
          gas_erg = gas_pre*igm1 + Ek/gas_dens;
        }
      }
    }
  }
  return;
}


void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  Real v2_phi, v2_theta, v3_phi, v3_theta;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v2_phi = 1.0; v2_theta = 0.0;
    v3_phi = 0.0; v3_theta = 1.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v2_phi = 0.0; v2_theta = 1.0;
    v3_phi = 1.0; v3_theta = 0.0;
  }

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) >= radius_outer_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_outer_damping)*inv_outer_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_r      = -1.5*(nu_alpha*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_theta  = 0.0;
          Real vel_gas_phi    = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi        -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_r;
          Real gas_vel2_0 = v2_phi*vel_gas_phi + v2_theta*vel_gas_theta;
          Real gas_vel3_0 = v3_phi*vel_gas_phi + v3_theta*vel_gas_theta;
          Real gas_pre_0  = cs_square_0*gas_rho_0;

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);
          Real &gas_pre  = prim(IPR, k, j, i);

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_0  - gas_pre )*R_func(i)*inv_damping_tau(i)*dt;

          // for damping zone, one degree of freedom needs to be left for the hydro system to relax itself.
          // This is purely based on experience.
          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            delta_gas_vel2 = 0.0;
          } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            delta_gas_vel3 = 0.0;
          }

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;
          gas_pre  += delta_gas_pre;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
          gas_erg = gas_pre*igm1 + Ek/gas_dens;
        }
      }
    }
  }
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {

        user_out_var(0,k,j,i) =
            phydro->flux[X1DIR](IDN,k,j,i);

        user_out_var(1,k,j,i) =
            phydro->flux[X1DIR](IDN,k,j,i+1);

        // 径向动量通量
        user_out_var(2,k,j,i) =
            phydro->flux[X1DIR](IM1,k,j,i);

        user_out_var(3,k,j,i) =
            phydro->flux[X1DIR](IM1,k,j,i+1);

        // 方位动量通量
        user_out_var(4,k,j,i) =
            phydro->flux[X1DIR](IM2,k,j,i);

        user_out_var(5,k,j,i) =
            phydro->flux[X1DIR](IM2,k,j,i+1);

      }
    }
  }
}

void MeshBlock::UserWorkInLoop() {

  Real &time = pmy_mesh->time;
  Real &dt   = pmy_mesh->dt;
  Real mygam = peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);
  int dk     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - NGHOST; int ju = je + NGHOST;
  int il = is - NGHOST; int iu = ie + NGHOST;

  if (Isothermal_Flag)
    LocalIsothermalEOS(this, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);

  if (Damping_Flag) {
    InnerWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
    OuterWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
  }

  return;
}
