import platform

import nox

if platform.machine() == "ARM64" and platform.system() == "Windows":
    python = ["3.13", "3.13t"]
else:
    python = ["3.9", "3.13", "3.13t", "pypy3.11"]


@nox.session(
    python=python,
    venv_backend="uv",
)
def test(session: nox.Session) -> None:
    session.install(".[test]")
    session.run("pytest", "-v", *session.posargs)
