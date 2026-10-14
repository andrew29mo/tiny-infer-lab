"""Check tracked files and reachable Git blobs for accidental private content.

This is a targeted guard, not an exhaustive secret detector or security audit.
It deliberately does not print matched secret values.
"""
import re
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "personal home path": re.compile(rb"/(?:Users|home)/[^/\s\"']+/"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "URL credential": re.compile(rb"https?://[^\s/:]+:[^\s/@]+@"),
    "private key": re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
}
FORBIDDEN_PARTS = {".aws", ".ssh", ".codex", ".agents", "__pycache__", "codex-notes", "backups"}
FORBIDDEN_NAMES = {'.netrc', '_netrc', '.npmrc', '.pypirc', '.git-credentials', '.gitconfig',
                   '.ds_store', 'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519',
                   'credentials.json', 'token.json', 'application_default_credentials.json'}
BUILD_SUFFIXES = {'.o', '.obj', '.a', '.so', '.dylib', '.dll', '.exe', '.class', '.wasm',
                  '.zip', '.tar', '.tgz', '.7z'}
BINARY_MAGIC = (b'MZ', b'!<arch>\n', b'\x00asm', b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe")


def findings(name, data):
    path = PurePosixPath(name)
    issues = []
    if (any(part in FORBIDDEN_PARTS for part in path.parts) or path.name == "AGENTS.md"
            or path.parts[0] == "build" or (path.name.startswith(".env") and path.name != ".env.example")
            or path.name.lower() in FORBIDDEN_NAMES or path.suffix.lower() in BUILD_SUFFIXES
            or path.name.lower().endswith((".bak", ".swp", "~"))
            or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx", ".pyc"}
            or (path.parts[0] == "results" and (len(path.parts) < 2 or path.parts[1] != "published"))):
        issues.append("local-only or credential file")
    if data.startswith(BINARY_MAGIC):
        issues.append("compiled binary")
    issues.extend(label for label, pattern in RULES.items() if pattern.search(data))
    return issues


def check(root=ROOT):
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    failures, count = [], 0
    for name in filter(None, tracked):
        path = root / name
        if path.is_symlink():
            failures.append(("worktree", name, "tracked symlink requires explicit review"))
            continue
        for issue in findings(name, path.read_bytes()):
            failures.append(("worktree", name, issue))
        count += 1
    names = {}
    revisions = subprocess.check_output(["git", "rev-list", "--all"], cwd=root, text=True).splitlines()
    for revision in revisions:
        tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision], cwd=root)
        for entry in filter(None, tree.split(b"\0")):
            header, name = entry.split(b"\t", 1)
            mode, kind, oid = header.decode().split()
            if kind != "blob":
                continue
            name = name.decode()
            names.setdefault(oid, set()).add(name)
            if mode == "120000":
                failures.append((revision[:12], name, "historical symlink requires explicit review"))
    # Batch-read once, including files removed from the current tree.
    result = subprocess.run(["git", "cat-file", "--batch"], cwd=root,
                            input=("\n".join(names) + "\n").encode(), capture_output=True, check=True)
    data, offset, blobs = result.stdout, 0, 0
    for oid, paths in names.items():
        end = data.index(b"\n", offset)
        header = data[offset:end].decode().split()
        size = int(header[2])
        body = data[end + 1:end + 1 + size]
        offset = end + size + 2
        if header[1] != "blob":
            continue
        blobs += 1
        for name in sorted(paths):
            for issue in findings(name, body):
                failures.append((oid[:12], name, issue))
    if failures:
        for scope, name, issue in failures:
            print("FAIL: %s %s: %s" % (scope, name, issue))
        raise SystemExit(1)
    print("PASS: %d tracked files and %d reachable historical blobs; no targeted private-content matches" % (count, blobs))
    print("Scope: home paths, known token/key patterns, local-only paths and compiled binaries; not an exhaustive secret audit")


if __name__ == "__main__":
    check()
