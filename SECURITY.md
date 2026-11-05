# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅ Yes     |
| < 2.0   | ❌ No      |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Use one of the following channels:

- **Private vulnerability report (preferred):** Use the
  [Report a vulnerability](https://github.com/The-Schultz-Lab/eagleplot/security/advisories/new)
  button on the Security tab. This keeps the report private until a fix is available.
- **Email:** Contact the maintainer directly at the address listed on the
  [NCCU Schultz Lab GitHub profile](https://github.com/The-Schultz-Lab).

## What to Include

A useful report includes:

- A description of the vulnerability and its potential impact
- Steps to reproduce (minimal example if possible)
- The version of EaglePlot affected
- Any suggested fix or mitigation

## Response Timeline

- **Acknowledgement:** within 5 business days
- **Assessment and triage:** within 2 weeks
- **Fix or workaround:** depends on severity; critical issues are prioritised

## Known Limitations by Design

The following behaviors are **intentional** and documented — please do not report
them as vulnerabilities unless you have found a bypass of the existing mitigations:

- **User-provided expressions in Analysis Tools and Data Tools.** Formula and
  curve-fit expressions are supplied as text and evaluated by EaglePlot.

  `eagleplot.data_tools.add_formula_column` evaluates them with an **AST whitelist**
  (`_safe_eval`): the expression is parsed, and only arithmetic, comparisons,
  `and`/`or`/`not`, conditional expressions, subscripting, literals, whitelisted
  names, and direct calls to the documented math functions are executed. Attribute
  access, imports, lambdas, comprehensions, assignments, and indirect calls are
  rejected. Exponents and expression length are bounded to prevent trivially
  constructed hangs.

  Row filtering (`filter_rows`) delegates to `pandas.DataFrame.eval`, which uses
  pandas' own restricted parser.

  **What this does and does not guarantee.** The whitelist is intended to prevent an
  expression from reaching the Python runtime — the filesystem, the process, or the
  import system. It is not a resource sandbox: an expression can still allocate large
  arrays or run for a long time. Treat it as a robust barrier against code execution,
  not as a guarantee of availability. If you find an expression that reaches attribute
  access, an import, or any callable outside the documented list, that **is** a
  vulnerability — please report it.

  Note that EaglePlot is designed to run on `localhost` for a single user. Binding the
  Streamlit GUI to a public interface exposes every input surface described below to
  anyone who can reach the port, and is not a supported configuration.

- **File uploads execute no code, by design.** Session files
  (Export → Load session) are decoded without `pickle`: arrays carry an explicit
  dtype, shape, and raw buffer, and anything not safely representable is refused
  rather than deserialized. Session files written by Plottle 2.0.1 and earlier used a
  pickle-based encoding; those entries are **refused on load**, not unpickled, and the
  loader reports what it skipped.

  `eagleplot.io.load_pickle` / `save_pickle` remain available in the **Python API**,
  where the caller has chosen to trust the file. `.pkl` is deliberately *not* accepted
  by the GUI's file uploader or batch folder import.

- **Local filesystem access from the GUI.** The Data Upload page's batch-folder
  import reads a user-supplied absolute path from the machine running the server. This
  is intended for local single-user use; do not expose it on a shared host.

## Scope

This policy covers the `eagleplot` package and its Streamlit GUI and CLI. It does not
cover third-party dependencies — please report those to their respective maintainers.
