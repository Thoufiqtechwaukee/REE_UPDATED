"""Publish deploy/dist to the static host over FTP.

Credentials come from the environment, never the command line, so they stay out
of shell history:

    FTP_HOST=68.178.174.167 FTP_USER=... FTP_PASS=... python deploy/ftp_deploy.py inspect
    FTP_HOST=68.178.174.167 FTP_USER=... FTP_PASS=... python deploy/ftp_deploy.py push

Commands:
    inspect   log in, show the directory tree, and guess the web root. Reads only.
    backup    download the current web root into deploy/backup-<timestamp>/.
    push      backup, then upload every file in deploy/dist to the web root.

The host is Microsoft FTP over plain port 21 - there is no FTPS on it, so the
password crosses the network in the clear. Rotate it once this is done.
"""

import ftplib
import os
import sys
from datetime import datetime
from pathlib import Path

DEPLOY_DIR = Path(__file__).parent
DIST_DIR = DEPLOY_DIR / "dist"

# Plesk on Windows usually serves from httpdocs; a plain IIS site from wwwroot.
WEB_ROOT_NAMES = ("httpdocs", "wwwroot", "public_html", "www", "web")


def connect() -> ftplib.FTP:
    host = os.environ.get("FTP_HOST", "").strip()
    user = os.environ.get("FTP_USER", "").strip()
    password = os.environ.get("FTP_PASS", "")
    missing = [n for n, v in (("FTP_HOST", host), ("FTP_USER", user), ("FTP_PASS", password)) if not v]
    if missing:
        sys.exit(f"error: set {', '.join(missing)} in the environment first")

    ftp = ftplib.FTP(host, timeout=60)
    ftp.login(user, password)
    ftp.set_pasv(True)
    print(f"connected: {ftp.getwelcome()}  as {user}")
    return ftp


def entries(ftp: ftplib.FTP, path: str = ".") -> list:
    """(name, is_dir, size) for one directory, via MLSD where the server has it."""
    try:
        return [(name, facts.get("type") == "dir", int(facts.get("size", 0)))
                for name, facts in ftp.mlsd(path)
                if name not in (".", "..")]
    except (ftplib.error_perm, ftplib.error_proto):
        out = []
        ftp.retrlines(f"LIST {path}", out.append)
        parsed = []
        for line in out:                      # IIS DOS-style listing
            bits = line.split(maxsplit=3)
            if len(bits) < 4:
                continue
            is_dir = bits[2].upper() == "<DIR>"
            size = 0 if is_dir else int(bits[2].replace(",", "") or 0)
            parsed.append((bits[3], is_dir, size))
        return parsed


def walk(ftp: ftplib.FTP, path: str, depth: int = 0, limit: int = 2) -> None:
    for name, is_dir, size in sorted(entries(ftp, path)):
        indent = "  " * depth
        print(f"  {indent}{name}{'/' if is_dir else ''}"
              f"{'' if is_dir else f'  ({size:,} bytes)'}")
        if is_dir and depth < limit:
            walk(ftp, f"{path.rstrip('/')}/{name}", depth + 1, limit)


def find_web_root(ftp: ftplib.FTP) -> str:
    """The directory the website is actually served from."""
    top = entries(ftp)
    names = {name for name, is_dir, _ in top if is_dir}
    for candidate in WEB_ROOT_NAMES:
        if candidate in names:
            return "/" + candidate
    if any(name.lower() == "index.html" for name, is_dir, _ in top if not is_dir):
        return "/"                            # already sitting in the web root
    return "/"


def download_tree(ftp: ftplib.FTP, remote: str, local: Path) -> int:
    local.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, is_dir, _ in entries(ftp, remote):
        remote_path = f"{remote.rstrip('/')}/{name}"
        if is_dir:
            count += download_tree(ftp, remote_path, local / name)
        else:
            with open(local / name, "wb") as fh:
                ftp.retrbinary(f"RETR {remote_path}", fh.write)
            count += 1
    return count


def backup(ftp: ftplib.FTP, web_root: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = DEPLOY_DIR / f"backup-{stamp}"
    print(f"\nbacking up {web_root} -> {target}")
    count = download_tree(ftp, web_root, target)
    print(f"  {count} files saved")
    return target


def push(ftp: ftplib.FTP, web_root: str) -> None:
    files = sorted(p for p in DIST_DIR.iterdir() if p.is_file())
    if not files:
        sys.exit(f"error: {DIST_DIR} is empty - run deploy/build_frontend.py first")

    print(f"\nuploading {len(files)} files to {web_root}")
    for path in files:
        with open(path, "rb") as fh:
            ftp.storbinary(f"STOR {web_root.rstrip('/')}/{path.name}", fh)
        print(f"  {path.name:14} {path.stat().st_size:>7,} bytes")


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    if command not in ("inspect", "backup", "push"):
        print(__doc__)
        return 2

    ftp = connect()
    try:
        web_root = find_web_root(ftp)
        print(f"web root looks like: {web_root}\n")
        print("remote tree:")
        walk(ftp, web_root)

        if command in ("backup", "push"):
            backup(ftp, web_root)
        if command == "push":
            push(ftp, web_root)
            print("\ndone - now load the site over http and check the console")
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
