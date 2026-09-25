"""Publish work as a GitHub release, with a rendered video and described assets.

Why this exists. The project convention is: after finishing any work, render the
artifacts to video, push them to a GitHub release, and annotate each artifact
with what it is. Until now that was done from throwaway scripts in /tmp, so it
was not repeatable and undescribed assets could slip through.

To make the convention hold, this tool enforces the part that is easy to skip:
**every published asset must carry a description.** A release cannot be created
without one, so "标注产物说明" is structural rather than a reminder.

Two subcommands:

    # render a mesh walkthrough (wraps tools.render_trajectory_video)
    python -m tools.publish_release render \
        --mesh outputs/vggt_slam_baseline/background_mesh_consensus.ply \
        --out deliverables/reports/foreground_removal/walkthrough_v3.mp4

    # publish assets, each with a description
    python -m tools.publish_release push \
        --tag foreground-removal-v3-20260924 \
        --title "去前景 v3：多视角 mask 一致性" \
        --body-file deliverables/reports/foreground_removal/REPORT_V3.md \
        --asset deliverables/reports/foreground_removal/compare_v3_consensus_3panel.mp4 \
            "主验收片：基线/v2/v3 三联" \
        --asset outputs/vggt_slam_baseline/background_mesh_consensus.ply \
            "v3 网格（620267 三角面 / 335146 顶点）"

The release body gets an asset table built from those descriptions, so the
description always appears on the release page itself, not only in the report.

Safety:
  * Videos are checked for faststart (moov before mdat) before upload -- a
    non-faststart mp4 makes the browser fetch nearly the whole file to start.
    It is remuxed losslessly (-c copy) first. Identity must then be checked with
    STREAM checksums, never the container checksum.
  * Uploads go to uploads.github.com, NOT api.github.com. Building the upload
    URL from the api base fails with "Name or service not known".
  * Re-running with --replace replaces same-named assets, so it is safe to
    repeat; a name collision without --replace is an error, not a silent
    overwrite.
  * Verification uses the server-reported sha256 digest, so a 27 MB asset does
    not need to be downloaded again (this link is unreliable for bulk pulls).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from vbr.cli import PROJECT_ROOT

API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"


def github_token() -> str:
    """Read the PAT from the git credential store (never from an argument).

    Taking it as a CLI flag would put it in shell history and process listings;
    the credential store is where git already keeps it.
    """
    candidates = [
        Path(os.path.expanduser("~/.git-credentials")),
        PROJECT_ROOT / ".git-credentials",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"https://([^:]+):([^@]+)@github\.com", line)
            if match:
                return match.group(2)
    raise SystemExit(
        "no github.com credential found in ~/.git-credentials; "
        "cannot publish without a token"
    )


def origin_repo() -> str:
    """owner/repo from the origin remote, tolerating a mirror prefix."""
    out = subprocess.run(["git", "remote", "get-url", "origin"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True)
    url = out.stdout.strip()
    match = re.search(r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?$", url)
    if not match:
        raise SystemExit(f"cannot parse owner/repo from origin: {url!r}")
    return match.group(1)


def request(url, token, method="GET", data=None, ctype=None, timeout=300):
    req = urllib.request.Request(
        url, method=method, data=data,
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "dsh-publish"})
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise SystemExit(f"{method} {url} -> {error.code}: {error.read()[:300]}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_offsets(path: Path):
    """(moov_offset, mdat_offset) by walking top-level MP4 boxes."""
    moov = mdat = None
    size = path.stat().st_size
    with open(path, "rb") as handle:
        offset = 0
        while offset < size:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) < 8:
                break
            length = int.from_bytes(header[:4], "big")
            kind = header[4:8]
            if kind not in (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"):
                break                      # not a top-level box; stop cleanly
            if length == 1:
                length = int.from_bytes(handle.read(8), "big")
            if length == 0:
                length = size - offset
            if kind == b"moov" and moov is None:
                moov = offset
            if kind == b"mdat" and mdat is None:
                mdat = offset
            offset += max(length, 8)
    return moov, mdat


def ensure_faststart(path: Path) -> bool:
    """Remux losslessly so moov precedes mdat. Returns True if it changed."""
    if path.suffix.lower() != ".mp4" or not path.exists():
        return False
    moov, mdat = box_offsets(path)
    if moov is not None and mdat is not None and moov < mdat:
        return False
    temporary = path.with_suffix(".faststart.mp4")
    subprocess.run(["/usr/bin/ffmpeg", "-y", "-v", "error", "-i", str(path),
                    "-c", "copy", "-movflags", "+faststart", str(temporary)],
                   check=True)
    temporary.replace(path)
    return True


def probe_video(path: Path) -> dict:
    out = subprocess.run(
        ["/usr/bin/ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,nb_frames",
         "-show_entries", "format=duration,size", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def cmd_render(args) -> None:
    """Render a walkthrough video (thin wrapper, so the convention is one call)."""
    # The underlying renderer locates the camera path via run_dir, and crashes
    # with an opaque TypeError if it is absent, so require one of them here.
    if not args.run_dir and not args.npz:
        raise SystemExit(
            "render needs --run-dir (e.g. outputs/vggt_slam_baseline) or an "
            "explicit --npz; the renderer derives the camera path from it")
    # sys.executable, not "python": the interpreter running this tool is the
    # project env, and bare `python` is not necessarily on PATH.
    command = [sys.executable, "-m", "tools.render_trajectory_video"]
    if args.run_dir:
        command += ["--run-dir", str(args.run_dir)]
    if args.mesh:
        command += ["--mesh", str(args.mesh)]
    if args.npz:
        command += ["--npz", str(args.npz)]
    if args.input_video:
        command += ["--input-video", str(args.input_video)]
    command += ["--out", str(args.out)]
    if args.limit:
        command += ["--limit", str(args.limit)]
    print("$", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    remuxed = ensure_faststart(args.out)
    info = probe_video(args.out)
    video = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    print(f"rendered {args.out}\n"
          f"  {video['width']}x{video['height']}  {video.get('nb_frames')} frames  "
          f"{args.out.stat().st_size/1e6:.1f} MB  "
          f"audio={'yes' if audio else 'no'}  "
          f"faststart={'remuxed' if remuxed else 'already'}\n"
          f"now describe it when publishing.")


def cmd_push(args) -> None:
    # Validate the request BEFORE acquiring anything (token, git remote), so a
    # user error is reported immediately and the release is never half-built.
    #
    # Every asset needs a description; that is the convention made structural.
    undescribed = [str(path) for path, text in args.asset if not text.strip()]
    if undescribed:
        raise SystemExit("these assets have no description:\n  "
                         + "\n  ".join(undescribed))

    for path, _ in args.asset:
        if not path.exists():
            raise SystemExit(f"asset not found: {path}")

    token = github_token()
    repo = args.repo or origin_repo()

    for path, _ in args.asset:
        if ensure_faststart(path):
            print(f"  remuxed {path.name} for faststart")

    print(f"repo {repo}, tag {args.tag}, {len(args.asset)} asset(s)")
    for path, description in args.asset:
        print(f"  {path.name:44s} {path.stat().st_size/1e6:7.2f} MB  {description}")

    body = args.body_file.read_text(encoding="utf-8") if args.body_file else ""
    table = ["", "## 产物说明", "",
             "| 文件 | 大小 | 说明 |", "| --- | --- | --- |"]
    for path, description in args.asset:
        table.append(f"| `{path.name}` | {path.stat().st_size/1e6:.2f} MB "
                     f"| {description} |")
    full_body = (body.rstrip() + "\n" + "\n".join(table) + "\n").lstrip()

    release = request(f"{API}/repos/{repo}/releases/tags/{args.tag}", token)
    if release is None:
        if request(f"{API}/repos/{repo}/git/ref/tags/{args.tag}", token) is None:
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                  capture_output=True, text=True).stdout.strip()
            request(f"{API}/repos/{repo}/git/refs", token, "POST", json.dumps(
                {"ref": f"refs/tags/{args.tag}", "sha": head}).encode(),
                "application/json")
            print(f"created tag {args.tag} at {head[:8]}")
        release = request(f"{API}/repos/{repo}/releases", token, "POST",
                          json.dumps({"tag_name": args.tag,
                                      "name": args.title or args.tag,
                                      "body": full_body, "draft": False,
                                      "prerelease": False}).encode(),
                          "application/json")
        print(f"created release {args.tag}")
    else:
        release = request(f"{API}/repos/{repo}/releases/{release['id']}", token,
                          "PATCH", json.dumps(
                              {"body": full_body,
                               "name": args.title or args.tag}).encode(),
                          "application/json")
        print(f"updated release body for {args.tag}")

    existing = {asset["name"]: asset["id"] for asset in release.get("assets", [])}
    for path, _ in args.asset:
        if path.name in existing:
            if not args.replace:
                raise SystemExit(
                    f"{path.name} already exists in {args.tag}; "
                    f"pass --replace to overwrite it")
            request(f"{API}/repos/{repo}/releases/assets/{existing[path.name]}",
                    token, "DELETE")
            print(f"  replaced existing {path.name}")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        url = (f"{UPLOADS}/repos/{repo}/releases/{release['id']}"
               f"/assets?name={path.name}")
        uploaded = request(url, token, "POST", path.read_bytes(), ctype,
                           timeout=1800)
        print(f"  uploaded {path.name} ({uploaded['size']/1e6:.1f} MB)")

    # Verify against the server's own digest: no re-download needed (this link
    # is unreliable for bulk transfers), and it proves the stored bytes match.
    final = request(f"{API}/repos/{repo}/releases/{release['id']}", token)
    local = {path.name: sha256(path) for path, _ in args.asset}
    print(f"\n=== {args.tag} ===")
    print(final["html_url"])
    verified = True
    for asset in final["assets"]:
        digest = (asset.get("digest") or "").removeprefix("sha256:")
        expected = local.get(asset["name"])
        state = ("not-checked" if expected is None or not digest
                 else ("MATCH" if digest == expected else "MISMATCH"))
        if state == "MISMATCH":
            verified = False
        print(f"  {asset['size']:>10}  {asset['name']:44s} sha256 {state}")
    if not verified:
        raise SystemExit("a published asset does not match the local file")
    print("published and verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a described release.")
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render a walkthrough video")
    render.add_argument("--run-dir", type=Path, default=None)
    render.add_argument("--mesh", type=Path, default=None)
    render.add_argument("--npz", type=Path, default=None)
    render.add_argument("--input-video", default=None)
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--limit", type=int, default=0)
    render.set_defaults(func=cmd_render)

    push = sub.add_parser("push", help="create/update a release with described assets")
    push.add_argument("--tag", required=True)
    push.add_argument("--title", default=None)
    push.add_argument("--body-file", type=Path, default=None,
                      help="markdown for the release body; an asset table with "
                           "the descriptions is appended automatically")
    push.add_argument("--asset", nargs=2, action="append", required=True,
                      metavar=("PATH", "DESCRIPTION"),
                      help="repeatable: an artifact and what it is")
    push.add_argument("--repo", default=None, help="override owner/repo")
    push.add_argument("--replace", action="store_true",
                      help="overwrite same-named assets instead of failing")
    push.set_defaults(func=cmd_push)

    args = parser.parse_args()
    if args.command == "push":
        # Plain strings: a shared type= would coerce descriptions to Path too.
        args.asset = [(Path(path), str(description))
                      for path, description in args.asset]
    args.func(args)


if __name__ == "__main__":
    main()