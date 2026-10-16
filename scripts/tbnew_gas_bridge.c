/* DSH's gas-only adapter. Download upstream TBnew/Verner sources separately.
 * This file contains no upstream model implementation. No grain-shielding or
 * XSPEC abundance lookup is permitted through this adapter.
 */
#include <math.h>
#include <stdlib.h>

extern double photabs(int index, double energy_ev);
extern double vernabs(double energy_ev, int atomic_number);
extern double h2abs(double energy_ev);
extern void tbnew(const double *, int, const double *, int, double *, double *, const char *);

static const int elements[18] = {1,2,6,7,8,10,11,12,13,14,16,17,18,20,24,26,27,28};

/* Forbidden dependencies: parameters below use absolute element columns,
 * and rho=0 disables the upstream grain implementation completely. */
float FGABND(char *element) { (void)element; abort(); }
double dgami_(double *x, double *y) { (void)x; (void)y; abort(); }

double dsh_atomic_sigma(int z, double energy_kev) {
    if (z < 1 || z > 30 || !isfinite(energy_kev) || energy_kev < 2 || energy_kev > 10)
        return NAN;
    for (int i=0; i<18; ++i)
        if (elements[i] == z) return photabs(i, energy_kev*1000.0);
    /* Extend the same published Verner basis to atoms not exposed by TBnew. */
    return vernabs(energy_kev*1000.0, z);
}

double dsh_h2_sigma(double energy_kev) { return h2abs(energy_kev*1000.0); }

double dsh_tbnew_gas_transmission(double low_kev, double high_kev,
        double nh_total, double fraction_h2, const double *residual_atoms_per_h) {
    double p[42] = {0};
    double gas_h = residual_atoms_per_h[0];
    if (!(gas_h > 0) || fraction_h2 < 0 || fraction_h2 > gas_h) return NAN;
    p[0] = nh_total * gas_h / 1e22;
    for (int i=1; i<18; ++i) p[i] = -nh_total * residual_atoms_per_h[elements[i]-1];
    p[18] = fraction_h2 / gas_h;
    /* p[19]=rho=0; all gas abundances are already residual, not total. */
    double bounds[2] = {low_kev, high_kev};
    double transmission, error;
    tbnew(bounds, 1, p, 0, &transmission, &error, "");
    return transmission;
}
