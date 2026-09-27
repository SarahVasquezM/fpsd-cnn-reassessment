#!/usr/bin/env python
"""Optional, disabled-by-default upload of small results to a non-main branch."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

OWNER = "SarahVasquezM"
REPOSITORY = "fpsd-cnn-reassessment"
BRANCH = "kaggle/week1-results"


def request(token: str, method: str, path: str, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(
        f"https://api.github.com/repos/{OWNER}/{REPOSITORY}{path}", data=body, method=method,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
    )
    with urlopen(req) as response:
        return json.load(response)


def main() -> None:
    if os.getenv("ENABLE_GITHUB_SYNC") != "1":
        raise SystemExit("Sincronización desactivada. Use ENABLE_GITHUB_SYNC=1 sólo después de revisar el bundle reducido.")
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Falta el secreto GITHUB_TOKEN")
    root = Path("/kaggle/working/github_results")
    if not root.is_dir():
        raise SystemExit("Descomprima github_results_bundle.zip en /kaggle/working/github_results")
    main_ref = request(token, "GET", "/git/ref/heads/main")
    encoded_branch = quote(BRANCH, safe="")
    try:
        request(token, "GET", f"/git/ref/heads/{encoded_branch}")
    except HTTPError as error:
        if error.code != 404:
            raise
        request(token, "POST", "/git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": main_ref["object"]["sha"]})
    for file in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = file.relative_to(root).as_posix()
        encoded = base64.b64encode(file.read_bytes()).decode()
        existing_sha = None
        try:
            existing_sha = request(token, "GET", f"/contents/{quote(relative)}?ref={encoded_branch}").get("sha")
        except HTTPError as error:
            if error.code != 404:
                raise
        payload = {"message": f"Add Kaggle week 1 result: {relative}", "content": encoded, "branch": BRANCH}
        if existing_sha:
            payload["sha"] = existing_sha
        request(token, "PUT", f"/contents/{quote(relative)}", payload)
    print(f"Resultados pequeños enviados a la rama {BRANCH}. El token no se imprimió ni se escribió en archivos.")


if __name__ == "__main__":
    main()
