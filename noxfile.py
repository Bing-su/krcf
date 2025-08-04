import nox


@nox.session(
    python=["3.9", "3.13", "3.13t", "pypy3.11"],
    venv_backend="uv",
)
def test(session: nox.Session) -> None:
    session.install(".[test]")
    session.run("pytest", "-v", *session.posargs)
