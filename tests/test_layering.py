from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from crucible import jobenv
from crucible.jobs import ALL_JOB_TYPES, CAPABILITIES

REPO = Path(__file__).resolve().parents[1]


def _modules_after_importing(module: str) -> set[str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import json, sys, {module}; print(json.dumps(sorted(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize(
    "module",
    [
        "crucible.jobs",
        "crucible.tasks",
        "crucible.installonsubmit",
        "crucible.api.routes.capability",
    ],
)
def test_the_server_side_never_imports_the_cli(module: str) -> None:
    loaded = _modules_after_importing(module)
    reached = sorted(
        name for name in loaded if name == "crucible.cli" or name.startswith("crucible.cli.")
    )
    assert not reached, f"importing {module} pulled in {reached}"


def test_leases_do_not_import_residency() -> None:
    assert "crucible.residency" not in _modules_after_importing("crucible.leases")


def test_every_job_capability_has_an_installer_that_is_installable() -> None:
    for capability in CAPABILITIES - {"echo"}:
        assert jobenv.INSTALLER_FOR.get(capability) in jobenv.INSTALLABLE_JOB_TYPES, capability


def test_every_installable_name_is_a_capability_the_registry_knows() -> None:
    assert set(jobenv.INSTALLABLE_JOB_TYPES) <= set(ALL_JOB_TYPES.values())


def test_every_worker_env_serves_job_types_the_registry_knows() -> None:
    served = {t for types in jobenv.JOB_TYPES_SERVED_BY_ENV.values() for t in types}
    assert served <= set(ALL_JOB_TYPES)


def test_a_job_type_with_no_installer_names_the_one_that_builds_it() -> None:
    assert jobenv.no_installer("rvc") is None
    shared = jobenv.no_installer("denoise")
    assert shared is not None and shared.shared_with == "rvc"
    unknown = jobenv.no_installer("sorcery")
    assert unknown is not None and unknown.shared_with is None
    assert "this build installs" in unknown.words


def test_a_job_names_itself_as_what_holds_the_card(tmp_path: Path) -> None:
    from crucible.jobs.base import Job
    from crucible.jobs.queue import busy_details

    job = Job(
        id="j1", type="asr", model="m", params={}, dir=tmp_path, created="t0", client="app"
    )
    assert job.busy_details()["since"] == "t0"
    assert busy_details(job) == job.busy_details()


def test_journal_identity_is_an_optional_member_of_the_protocol() -> None:
    from crucible.jobs import _JOB_TYPE_MEMBERS
    from crucible.jobs.asr import AsrJobType
    from crucible.jobs.base import OPTIONAL_JOB_TYPE_MEMBERS, JobType
    from crucible.jobs.echo import EchoJobType

    assert "journal_identity" in OPTIONAL_JOB_TYPE_MEMBERS
    assert "journal_identity" not in _JOB_TYPE_MEMBERS
    assert JobType.journal_identity is None
    assert getattr(EchoJobType(), "journal_identity", None) is None
    assert callable(AsrJobType.journal_identity)
