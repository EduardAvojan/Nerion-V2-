

from __future__ import annotations
import click
from selfcoder.security import scanner, rules, report

@click.command("scan")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--fail-on", type=click.Choice(["low", "medium", "high", "critical"], case_sensitive=False), help="Fail if findings at or above this severity are found")
@click.option("--json-out", type=click.Path(dir_okay=False, writable=True), help="Write JSON report to file")
def cli(paths: tuple[str, ...], fail_on: str | None, json_out: str | None) -> None:
    """
    Run Nerion security scanner on given paths.
    """
    findings = scanner.scan(paths or ["."], rules.DEFAULT_RULES)
    if json_out:
        report.write_json(findings, json_out)

    # Print results
    report.print_findings(findings)

    # Enforce fail-on policy
    if fail_on and report.has_severity(findings, fail_on):
        raise SystemExit(1)
    raise SystemExit(0)