"""CLI help 查询测试：所有命令与子命令均可查询用法。"""

from typer.testing import CliRunner

from assistlabel.cli import app

runner = CliRunner()


def test_top_level_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("init", "run", "status", "export", "validate", "verify", "models", "help"):
        assert cmd in r.output


def test_help_command_lists_subcommands():
    r = runner.invoke(app, ["help"])
    assert r.exit_code == 0
    assert "run" in r.output and "models" in r.output


def test_help_single_command():
    r = runner.invoke(app, ["help", "run"])
    assert r.exit_code == 0
    assert "--limit" in r.output and "--redo-detect" in r.output


def test_help_multi_level():
    r = runner.invoke(app, ["help", "models", "download"])
    assert r.exit_code == 0
    assert "--source" in r.output


def test_help_unknown_command_exits_2():
    r = runner.invoke(app, ["help", "nope"])
    assert r.exit_code == 2


def test_every_command_supports_dash_dash_help():
    for argv in (["run"], ["status"], ["export"], ["validate"], ["verify"], ["models"], ["models", "list"]):
        r = runner.invoke(app, [*argv, "--help"])
        assert r.exit_code == 0, f"{argv} --help 失败"
