import nox

python = ["3.9", "3.14", "3.14t", "pypy3.11"]
venv_backend = "uv"


@nox.session(
    python=python,
    venv_backend=venv_backend,
)
def test(session: nox.Session) -> None:
    session.install(".[test]")
    session.run("pytest", "-v", *session.posargs)
