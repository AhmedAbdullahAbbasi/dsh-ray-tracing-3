# Stage 1: actual Windows end-to-end smoke validation

Run the delivery's `apply_and_run_windows_smoke.ps1` in PowerShell. It imports
the validation update into the existing Desktop checkout by fast-forward,
using its existing virtual environment. It refuses tracked local edits or an
unexpected branch/checkpoint. It does not reset, clean or force-push.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apply_and_run_windows_smoke.ps1 -Push
```

Omit `-Push` to keep the validation tooling local. For a different location,
add `-RepoPath "C:\path\to\dsh-simulation-config"`.

The portable underlying command is:

```powershell
.\.venv\Scripts\python.exe scripts\validate_configured_smoke.py
```

Every invocation creates a uniquely named session below `outputs/windows-smoke`.
It leaves existing simulation outputs and the two tracked example TOML files
intact. Generated TOML files explicitly select absolute input paths, fresh
outputs, linear/log plots and scored-event maps. Subprocesses run outside the
repository in a directory containing spaces, checking editable installation
and configuration-relative path handling.

## Cases and independent gates

1. The existing 64-packet legacy-material example must check and run cleanly.
2. The existing 64-packet declared-WD01 example must check and run cleanly.
3. A separate 128-packet legacy control with cap one must return exit 3, report
   numerical failures and write no production NPZ/FITS or completion marker.

The product inspector does not call production builders or the quality
evaluator. It verifies completion inventory hashes; original/resolved and
NPZ/FITS configuration identity; FITS checksums/datasums; packet accounting and
terminal states; axes, units and cube ordering; first/multiple-scatter closure;
source-fluence and cloud-column closure; all exported raw-moment arrays; NPZ/FITS
agreement; plot files; and scored-event maps. Zero-score images are allowed.
No threshold is tuned to make a particular random image pass. This gate makes
no statement about Monte Carlo convergence or detector realism.

`smoke_summary.json` records actual platform, Python/dependency versions,
per-case results, fluences and terminal counts. `smoke_report.zip` includes the
summary, logs, original/resolved settings, completion manifests and PNG views.
Large NPZ/FITS products remain in the session folder and are not duplicated in
the report ZIP. Send that report ZIP for review before proceeding to Stage 2.

Linux runs of this tool validate the automation and inspector only; Windows
execution is established only by a report recording `windows_execution: true`.
PowerShell execution itself must also be checked on the Windows machine.
