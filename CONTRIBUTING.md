# Contributing

Changes should preserve the contract in `include/tiny_infer.h` and include evidence
that tests exercise production objects. Run:

```sh
make test sanitize mutation build-check
make BLAS=1 test sanitize mutation
```

On macOS, BLAS builds require macOS 15+ with the Command Line Tools. On Linux,
install the system OpenBLAS development package first. The default build has no
BLAS dependency. Only macOS and Linux are currently supported by the build tools.

For benchmark changes, retain the exact workload, build flags, thread policy,
raw samples and independent numerical reference. Report regressions as well as
improvements. Do not turn shared CI timing into a pass/fail speed threshold.

The repository does not currently grant an open-source license. Contact the owner
before contributing code that would require a licensing agreement.
