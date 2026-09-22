# Stage 9B: controlled virtual-observer escape attenuation

Run from the repository root with the previously integrated material tables:

~~~powershell
python -m unittest tests.test_validation_escape -v
python -m scripts.run_escape_attenuation_validation --output validation_outputs/stage9b_escape_attenuation.json
~~~

The runner is deterministic and small: it **does not repeat** the 2.5-million
packet flare, require HEASOFT at runtime, or perform a statistical transport
simulation. Its default configuration checks both frozen 3.3/4.9/6.9 keV
materials and the co-registered 2–10 keV tables at 2, 3.3, 4.9, 5.35, 6.9,
and 10 keV. The 5.35-keV case tests interpolation between energy nodes.

## Independent reference

Each of two *fixed* physical histories has two dust scatter records at
observer distances 5 and 3 kpc, followed by an analog absorption at 1 kpc.
The histories use different off-axis sky cells and are not resimulated when
the material or cloud changes. For each virtual event, the ray to the
observer travels radially inward through four 2-kpc shell cells. A separate
NumPy reference sums each complete foreground cell column and the occupied
fraction of the event cell. It never calls the production ray integrator.
This is a closed-form reference for this controlled geometry, including
off-axis rays through different angular cells and more than one dusty shell.

The scorer runs on four cloud variants: vacuum, the baseline, an extra
foreground column on one sightline, and an extra column *behind* all
scatter events. Each is scored with the actual scattering table and either
actual TBabs absorption or absorption set to zero for the virtual escape
factor. A mode with scattering still present but absorption disabled is an
estimator control; it is not an alternative physical dust model.

For a fixed record with independently computed observer-leg column,

$$
T_{\rm escape}=\exp[-N_{\rm H,escape}
(\sigma_{\rm sca}(E)+\sigma_{\rm abs}(E))].
$$

The absolute score is compared to the source weight, source-to-event
inverse-square factor, launch importance factor, unchanged evaluated phase,
and the independently predicted escape transmission. Paired scores make
the phase and launch factors cancel, furnishing direct checks:

$$
\frac{w_{\rm extra\,foreground}}{w_{\rm base}}
=\exp[-\Delta N_{\rm H}(\sigma_{\rm sca}+\sigma_{\rm abs})],
\qquad
\frac{w_{\rm abs\,on}}{w_{\rm abs\,off}}
=\exp[-N_{\rm H,escape}\sigma_{\rm abs}].
$$

Adding column *behind* a scattering event must leave its virtual observer
score unchanged when its incident record is held fixed. This tests that
the scorer does not apply incoming-path extinction a second time. Both
earlier scattering orders must score even though the actual analog photon
later gets absorbed; the absorption record itself must not score.

## Passing and interpretation

Each saved energy case records scenario-wise column, optical-depth,
transmission and weight errors, plus paired-ratio errors and Boolean
checks. Columns must agree within \(8\times10^{-6}\) relative. Weights,
transmissions, and paired ratios must agree within \(2\times10^{-4}\)
relative; optical depths within \(5\times10^{-6}\) absolute. These are
deterministic numerical tolerances, not Monte Carlo error bars.
The report includes the git revision/status, environment, and material
checksums, and exits nonzero if any case fails.

An accepted Stage 9B validates *this fixed-history estimator* and its
extinction law over these shell/sightline/energy cases. The phase factor
in the absolute-weight reference is deliberately kept as an unchanged
nuisance; phase normalization has separate tests. Stage 9B does not
establish absorption-enabled multiple-scattering event frequencies, a
converged production image, uncertainty in the large four-cloud run,
or a correct telescope response. Stage 9C is the separate repeated
scattering test, followed by Stage 9D with physical absorption enabled.
