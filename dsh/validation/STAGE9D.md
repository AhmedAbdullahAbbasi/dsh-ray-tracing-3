# Stage 9D: repeated scattering with physical absorption

Run this local gate from the repository root after installing the patch:

~~~powershell
python -m unittest tests.test_validation_absorbing_multiple_scattering -v
python -m scripts.run_absorbing_multiple_scattering_validation --output validation_outputs/stage9d_absorbing_multiple_scattering.json
~~~

The default check samples 20,000 analog photons for each of ten material and
optical-depth cases: the three-energy v1 dust/TBabs bundle at 3.3 and 6.9 keV;
the matched 2–10 keV bundle at 2, 5.35, and 10 keV; and scattering depths 1.5
and 2.4. The 5.35-keV case checks material interpolation. In each case the
*physical* absorption cross section is present, without modifying its ratio
to scattering. The shell column is set by the target **scattering** depth.

## Independent conditional reference

Actual photons are launched through the constant-density 4–5 kpc radial shell
at a 10 kpc source distance. Each sampled scattering changes the subsequent
flight direction. From each *actual* flight start, a separate NumPy sphere
intersection supplies the dust column to the observer plane or world boundary.
This reference never calls the JAX voxel ray integrator. It predicts competing
outcomes independently for all four flight orders:

\[
P_{\rm sca}=(1-e^{-N_H(\sigma_{\rm sca}+\sigma_{\rm abs})})
 \frac{\sigma_{\rm sca}}{\sigma_{\rm sca}+\sigma_{\rm abs}},\qquad
P_{\rm abs}=(1-e^{-N_H(\sigma_{\rm sca}+\sigma_{\rm abs})})
 \frac{\sigma_{\rm abs}}{\sigma_{\rm sca}+\sigma_{\rm abs}},\qquad
P_{\rm exit}=e^{-N_H(\sigma_{\rm sca}+\sigma_{\rm abs})}.
\]

Predicted category counts are sums of individual conditional probabilities;
their variances are sums of \(p_i(1-p_i)\). This correctly accounts for
different path columns following scattering without treating terminal outcomes
as a simple global Poisson law. Conditional on any event, its independent
cumulative shell column must follow a truncated exponential distribution;
the report checks the 25%, 50%, and 75% quantiles. The code also checks the
scattering angle CDF, process branching, masked record slots, interaction
orders, elastic scatter, absorption energy deposit, zero post-absorption
momentum, and absorption after at least two scatterings.
The focused unit test also passes actual absorbed histories to the virtual
observer scorer. It verifies that both earlier scattering orders are still
scored, that absorption itself is masked, and that real-event escape columns
and transmissions agree with the separate sphere-chord reference.

A fourth physical scatter is recorded as `MAX_INTERACTIONS` by the controlled
four-slot validation run. This is a numerical censoring category, **not** a
physical escape or absorption. A real simulation needs a sufficiently large
interaction cap and must inspect its cap fraction independently.

Each category on each flight must have at least 20 expected events and its
residual must be within five standard deviations. The quantile and process
checks use the same five-sigma marginal limit. Correlations among the tests
mean this is not one familywise confidence level; if a marginal is weak or
close to its threshold, increase packets or repeat with another seed. The
machine-readable JSON records both material hashes, Git state, exact packet
settings, category expectations, variances, and observed counts.

This establishes absorption-enabled analog event histories in the controlled
shell and checks virtual scores on a small set of actual absorbed histories.
It does not establish convergence or physical plausibility of the
2.5-million-packet four-cloud flare images. Stage 9B separately checked
virtual-observer extinction on fixed scattering records; production-image
and convergence checks remain necessary before declaring the flare successful.
