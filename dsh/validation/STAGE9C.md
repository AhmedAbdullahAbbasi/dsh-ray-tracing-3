# Stage 9C: analog repeated scattering without absorption

Run locally from the repository root with the bundled scattering tables:

~~~powershell
python -m unittest tests.test_validation_multiple_scattering -v
python -m scripts.run_multiple_scattering_validation --output validation_outputs/stage9c_multiple_scattering.json
~~~

The default runner uses 20,000 photons per energy/optical-depth combination,
with actual NewDust scattering at 3.3 and 6.9 keV and the 2–10 keV table at
2, 5.35 and 10 keV. The 5.35-keV case is between tabulated energy nodes.
Each energy is checked at scattering optical depth 1.5 and 2.4 through a
constant-density 4–5 kpc spherical shell in a 10 kpc world. An absorption
array of zeros is passed explicitly; this is a controlled transport check,
not a physical prediction without photoelectric absorption.

## Independent reference

The reference uses NumPy sphere intersections to calculate the dust column
through the shell, and independently finds the observer-plane or outer-sphere
flight limit. For each *actual scattering record*, it conditions on that
record's outgoing position and direction and predicts the next flight's
probability of another scattering:

\[
P(\mathrm{scatter\ on\ next\ flight}\mid\mathrm{start})
 = 1-\exp[-\sigma_{\rm sca}(E)N_{\rm H}(\mathrm{start\ to\ boundary})].
\]

The prediction is made separately for all four flights, including flights
after the first, second, and third actual scatter. A scatter in the fourth
slot is classified as `MAX_INTERACTIONS`, never as a physical escape.
Each flight compares observed and summed predicted scatter counts with
variance \(\sum p_i(1-p_i)\). These conditional sums are appropriate when
starting points differ between photons, as the successive directions are
drawn from the *real* material angular table. Event positions are converted
to their independent cumulative column and tested against the conditional
truncated-exponential distribution at 25%, 50%, and 75%. Outgoing deflection
angles are compared to 50% and 90% phase-CDF quantiles. The report checks
elastic energy, analog event ordering, valid-slot masks, terminal outcomes,
and absence of absorption. Both material sets use their actual integrated
scattering cross sections, while the 2–10 case also tests energy interpolation.

Each reported marginal must be within five standard deviations and each
flight must have at least 20 expected scatters. These are exploratory
per-marginal gates, not a joint 5-sigma confidence level: the tests and
successive flights are correlated. If a row lacks adequate expected events,
the gate fails and needs more photons. Store the output JSON for comparison;
it records the table hash, Git state, configuration, and per-flight counts.

Passing Stage 9C validates repeated *pure-scattering* analog flight hazards,
locations, event records and phase sampling in this spherical-shell setup.
It does not establish absorption-enabled scattering-order fractions or a
statistically converged four-cloud flare image. Stage 9D will repeat the
order-resolved tests with physical absorption and competing processes.
