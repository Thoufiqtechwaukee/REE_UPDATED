"""Publish deploy/dist to the static host over FTP.

Credentials come from the environment, never the command line, so they stay out
of shell history:

    FTP_HOST=50.63.8.208 FTP_USER=... FTP_PASS=... python deploy/ftp_deploy.py inspect
    FTP_HOST=50.63.8.208 FTP_USER=... FTP_PASS=... python deploy/ftp_deploy.py push

Commands:
    inspect   log in, list the web root, and report what is there. Reads only.
    backup    download the current web root into deploy/backup-<timestamp>/.
    push      backup, then upload every file in deploy/dist to the web root.

Notes on this particular host. It is Microsoft FTP on plain port 21 at
50.63.8.208 - the web server itself. The 68.178.174.167 quoted alongside the
password also answers on 21 but refuses this account. There is no working FTPS,
so the password crosses the network in the clear; rotate it afterwards. MLSD is
not implemented, so listings come from the DOS-style LIST output. Worst of all
the passive data channel times out perhaps half the time, and once it does the
control connection is unusable - so every transfer runs through retrying(),
which reconnects from scratch on each attempt.
"""

import ftplib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

DEPLOY_DIR = Path(__file__).parent
DIST_DIR = DEPLOY_DIR / "dist"

TIMEOUT = 20          # a hung data channel should fail fast and be retried
ATTEMPTS = 8          # ... which needs a generous number of goes
WEB_ROOT_NAMES = ("httpdocs", "wwwroot", "public_html", "www", "web")


class Session:
    """An FTP connection that rebuilds itself whenever a transfer fails."""

    def __init__(self):
        self.host = os.environ.get("FTP_HOST", "").strip()
        self.user = os.environ.get("FTP_USER", "").strip()
        self.password = os.environ.get("FTP_PASS", "")
        missing = [name for name, value in
                   (("FTP_HOST", self.host), ("FTP_USER", self.user), ("FTP_PASS", self.password))
                   if not value]
        if missing:
            sys.exit(f"error: set {', '.join(missing)} in the environment first")
        self.ftp = None

    def connect(self) -> ftplib.FTP:
        self.close()
        try:
            ftp = ftplib.FTP(timeout=TIMEOUT)
            ftp.connect(self.host, 21)
            ftp.login(self.user, self.password)
        except ftplib.error_perm as exc:
            sys.exit(f"login refused for {self.user!r}: {exc}")
        except OSError as exc:
            sys.exit(f"could not reach {self.host}: {exc}")
        ftp.set_pasv(True)
        self.ftp = ftp
        return ftp

    def close(self) -> None:
        if self.ftp is not None:
            try:
                self.ftp.quit()
            except Exception:
                try:
                    self.ftp.close()
                except Exception:
                    pass
            self.ftp = None

    def run(self, action, what: str):
        """Run one data-channel operation, reconnecting between attempts.

        A timed-out passive transfer leaves the control connection wedged, so
        retrying on the same handle just fails again. Each attempt gets a fresh
        login, and a fresh PASV port with it - which is the thing that actually
        varies between a failure and a success here.
        """
        for attempt in range(1, ATTEMPTS + 1):
            try:
                return action(self.connect())
            except (TimeoutError, OSError, ftplib.error_temp, EOFError) as exc:
                if attempt == ATTEMPTS:
                    raise
                print(f"    {what}: {type(exc).__name__}, retry {attempt}/{ATTEMPTS - 1}")
                time.sleep(1.5)


def parse_listing(lines: list) -> list:
    """IIS DOS-style LIST output -> (name, is_dir, size)."""
    parsed = []
    for line in lines:
        bits = line.split(maxsplit=3)
        if len(bits) < 4:
            continue
        is_dir = bits[2].upper() == "<DIR>"
        size = 0 if is_dir else int(bits[2].replace(",", "") or 0)
        parsed.append((bits[3], is_dir, size))
    return parsed


def entries(session: Session, path: str = "/") -> list:
    def action(ftp):
        lines = []
        ftp.retrlines(f"LIST {path}", lines.append)
        return parse_listing(lines)

    return session.run(action, f"LIST {path}")


def find_web_root(session: Session) -> str:
    top = entries(session, "/")
    dirs = {name for name, is_dir, _ in top if is_dir}
    for candidate in WEB_ROOT_NAMES:
        if candidate in dirs:
            return "/" + candidate
    return "/"                     # index.html sits at the login directory


def download_tree(session: Session, remote: str, local: Path) -> int:
    local.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, is_dir, _ in entries(session, remote):
        remote_path = f"{remote.rstrip('/')}/{name}"
        if is_dir:
            count += download_tree(session, remote_path, local / name)
            continue

        def action(ftp, target=local / name, source=remote_path):
            with open(target, "wb") as fh:
                ftp.retrbinary(f"RETR {source}", fh.write)

        session.run(action, f"RETR {remote_path}")
        print(f"    saved {remote_path}")
        count += 1
    return count


def backup(session: Session, web_root: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = DEPLOY_DIR / f"backup-{stamp}"
    print(f"\nbacking up {web_root} -> {target.name}/")
    count = download_tree(session, web_root, target)
    print(f"  {count} files saved to {target}")
    return target


def push(session: Session, web_root: str) -> None:
    files = sorted(p for p in DIST_DIR.iterdir() if p.is_file())
    if not files:
        sys.exit(f"error: {DIST_DIR} is empty - run deploy/build_frontend.py first")

    print(f"\nuploading {len(files)} files to {web_root}")
    for path in files:
        def action(ftp, source=path):
            with open(source, "rb") as fh:
                ftp.storbinary(f"STOR {web_root.rstrip('/')}/{source.name}", fh)

        session.run(action, f"STOR {path.name}")
        print(f"  {path.name:14} {path.stat().st_size:>7,} bytes")


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    if command not in ("inspect", "backup", "push"):
        print(__doc__)
        return 2

    session = Session()
    try:
        web_root = find_web_root(session)
        print(f"connected to {session.host} as {session.user}")
        print(f"web root: {web_root}\n")
        for name, is_dir, size in sorted(entries(session, web_root)):
            print(f"  {name}{'/' if is_dir else ''}"
                  f"{'' if is_dir else f'  ({size:,} bytes)'}")

        if command in ("backup", "push"):
            backup(session, web_root)
        if command == "push":
            push(session, web_root)
            print("\ndone - load the site over http and check the browser console")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
