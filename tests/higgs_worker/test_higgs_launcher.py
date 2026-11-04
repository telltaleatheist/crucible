"""narrator's OWN Higgs v3 launcher: the packaged pair, and what it refuses.

WHY THIS FILE EXISTS. Until 2026-09-13 `serve_higgs_v3.sh` lived only in
BookForge's `electron/scripts/higgs/`, and `HiggsV3ServedBackend` could be
built only by a caller that knew where a BookForge checkout was. Crucible's
first real `tts` render is what found that: narrator exited 3 before `ready`,
and behind the missing `HIGGS_STACK` sat a launcher no other client could name.

The script now ships INSIDE the package
(`narrator/engine/higgs/launch/serve_higgs_v3.sh`, with the certified
frames-7500 deploy profile beside it) and is resolved through
`importlib.resources`. What is asserted here:

  * THE PAIR SHIPS TOGETHER and the profile is the CERTIFIED BYTES. The script
    reads the profile as `$(dirname "$0")/...`, so one without the other is a
    server that truncates every long chunk at 81.92 s and says nothing.
  * THE THREE LAUNCHER MODES - attach, the operator's override, narrator's own -
    and the refusal for an override that is not there.
  * THE SCRIPT'S OWN two new contracts, by RUNNING IT: `HIGGS_ENV` is required
    by name, and an unset `HIGGS_DEPLOY_CONFIG` means the certified sibling
    while an explicitly EMPTY one means vllm-omni's auto-discovered profile.

The script is run against a FAKE env prefix whose `bin/vllm-omni` echoes its
argv, so nothing here needs a GPU, a model or vllm-omni.
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine.higgs import v3_served                          # noqa: E402
from crucible_higgs.engine.higgs.v3_served import (HiggsV3ServedBackend,   # noqa: E402
                                             LAUNCHER_ATTACH,
                                             LAUNCHER_OPERATOR,
                                             LAUNCHER_PACKAGED)

#: THE CERTIFIED DEPLOY PROFILE, by sha256. The training side's cap
#: certificates were measured against these exact bytes, and `.gitattributes`
#: pins the file to LF for that reason: YAML parses either way, so a CRLF
#: checkout would deploy a profile that is not the one anything was certified
#: against and NOTHING would fail. This is what notices.
DEPLOY_PROFILE_SHA256 = (
    '24d288f193eaa8c5c10387d890b648949b88e8d357a3779e8c9487f9d38c7481')


def _bash():
    found = shutil.which('bash')
    if not found:
        raise unittest.SkipTest(
            'no bash on this machine; narrator launches the Higgs v3 server '
            'with `bash <script>` (inside the WSL guest on the Windows arm), so '
            "this machine could not start one anyway and the script's own "
            'refusals cannot be exercised here.')
    return found


class PackagedLauncherTest(unittest.TestCase):
    """The two files, and that they are the ones that were certified."""

    def test_the_script_and_its_profile_ship_in_one_directory(self):
        script = v3_served.packaged_serve_script()
        self.assertTrue(os.path.isfile(script), script)
        profile = os.path.join(os.path.dirname(script),
                               v3_served.PACKAGED_DEPLOY_CONFIG)
        self.assertTrue(os.path.isfile(profile), profile)

    def test_the_deploy_profile_is_the_certified_bytes(self):
        profile = os.path.join(
            os.path.dirname(v3_served.packaged_serve_script()),
            v3_served.PACKAGED_DEPLOY_CONFIG)
        with open(profile, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        self.assertEqual(
            digest, DEPLOY_PROFILE_SHA256,
            'the shipped deploy profile is not the certified file. Either the '
            'checkout rewrote its line endings (.gitattributes pins it to LF) '
            'or the profile changed, in which case every cap certificate bound '
            'to it has to be re-measured.')

    def test_it_is_resolved_once_and_kept(self):
        """The path is handed to bash minutes after it is resolved, from
        another method, so the ExitStack that `as_file` needs for a zip install
        must outlive the call. Same object twice is how that is visible."""
        self.assertEqual(v3_served.packaged_serve_script(),
                         v3_served.packaged_serve_script())


class LauncherModeTest(unittest.TestCase):
    """Three modes, one decision, made in `__init__` and nowhere else."""

    def setUp(self):
        # RESTORED WHETHER IT WAS SET OR NOT. A cleanup registered only for a
        # variable that already existed leaves whatever the TEST sets behind,
        # and a leaked NARRATOR_HIGGS3_SERVE_SCRIPT turns every later
        # attach-mode test in the process into a launch (measured: 16 failures
        # in the full suite that all passed file-by-file).
        for name in (v3_served.BASE_URL_ENV, v3_served.SERVE_SCRIPT_ENV,
                     v3_served.SERVE_MAX_NUM_SEQS_ENV):
            self._restore(name)
            os.environ.pop(name, None)
        # A LAUNCHING backend states the server's admission width; that is a
        # separate contract (`serve_concurrency`) with its own refusal.
        os.environ[v3_served.SERVE_MAX_NUM_SEQS_ENV] = '16'
        self.dir = tempfile.mkdtemp(prefix='narrator-H-modes-')
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _restore(self, name):
        previous = os.environ.get(name)
        if previous is None:
            self.addCleanup(os.environ.pop, name, None)
        else:
            self.addCleanup(os.environ.__setitem__, name, previous)

    def _script(self, name='serve_v3.sh'):
        path = os.path.join(self.dir, name)
        with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write('#!/bin/bash\nexit 0\n')
        os.chmod(path, 0o755)
        return path

    def test_no_variable_at_all_runs_narrators_own(self):
        backend = HiggsV3ServedBackend()
        self.addCleanup(backend._close_log)
        self.assertEqual(backend.launcher_source, LAUNCHER_PACKAGED)
        self.assertEqual(backend.serve_script,
                         v3_served.packaged_serve_script())

    def test_the_env_override_wins_over_the_packaged_one(self):
        mine = self._script()
        os.environ[v3_served.SERVE_SCRIPT_ENV] = mine
        backend = HiggsV3ServedBackend()
        self.addCleanup(backend._close_log)
        self.assertEqual(backend.launcher_source, LAUNCHER_OPERATOR)
        self.assertEqual(backend.serve_script, mine)

    def test_a_base_url_is_attach_and_has_no_launcher(self):
        backend = HiggsV3ServedBackend(base_url='http://127.0.0.1:9999')
        self.addCleanup(backend._close_log)
        self.assertEqual(backend.launcher_source, LAUNCHER_ATTACH)
        self.assertFalse(backend.serve_script)
        with self.assertRaises(ValueError):
            backend.launch_command()

    def test_an_override_that_is_not_there_is_refused_not_replaced(self):
        """Substituting the packaged script for a path an operator named would
        start a server with different flags and report success."""
        missing = os.path.join(self.dir, 'gone.sh')
        os.environ[v3_served.SERVE_SCRIPT_ENV] = missing
        with self.assertRaises(ValueError) as caught:
            HiggsV3ServedBackend()
        self.assertIn(v3_served.SERVE_SCRIPT_ENV, str(caught.exception))
        self.assertIn(repr(missing), str(caught.exception))

    @unittest.skipUnless(sys.platform == 'win32', 'the Windows arm only')
    def test_a_guest_path_on_windows_is_not_checked(self):
        """BookForge names the launcher as the DISTRO sees it. That file is not
        on the Windows filesystem, so checking it here would refuse a path that
        exists."""
        os.environ[v3_served.SERVE_SCRIPT_ENV] = (
            '/home/<user>/anaconda3/envs/higgs3/bin/serve_higgs_v3.sh')
        backend = HiggsV3ServedBackend()
        self.addCleanup(backend._close_log)
        self.assertEqual(backend.launcher_source, LAUNCHER_OPERATOR)
        os.environ[v3_served.SERVE_SCRIPT_ENV] = (
            r'\\wsl$\Ubuntu\home\<user>\serve_higgs_v3.sh')
        second = HiggsV3ServedBackend()
        self.addCleanup(second._close_log)
        self.assertEqual(second.launcher_source, LAUNCHER_OPERATOR)


class TheScriptItselfTest(unittest.TestCase):
    """RUN the packaged script, against a fake env prefix. No GPU, no model.

    The script `exec`s `$HIGGS_ENV/bin/vllm-omni`, so a fake one that echoes
    its argv turns the whole launch line into readable output - which is how
    the deploy-profile default is asserted as BEHAVIOUR rather than as a line
    of shell somebody read.
    """

    def setUp(self):
        self.bash = _bash()
        self.script = v3_served.packaged_serve_script()
        self.dir = tempfile.mkdtemp(prefix='narrator-H-script-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.prefix = os.path.join(self.dir, 'env')
        os.makedirs(os.path.join(self.prefix, 'bin'))
        fake = os.path.join(self.prefix, 'bin', 'vllm-omni')
        with io.open(fake, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write('#!/bin/bash\necho "ARGV: $*"\n')
        os.chmod(fake, 0o755)
        # A MERGED CHECKPOINT DIRECTORY, so the script never looks in the HF
        # cache (which this machine may or may not have) and never refuses for
        # a reason this test is not about.
        self.model = os.path.join(self.dir, 'merged')
        os.makedirs(self.model)

    def run_script(self, **overrides):
        environment = dict(os.environ)
        environment.pop('HIGGS_ENV', None)
        environment.pop('HIGGS_DEPLOY_CONFIG', None)
        environment['HIGGS_MODEL_DIR'] = self.model
        for key, value in overrides.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        return subprocess.run(
            [self.bash, self.script], env=environment, capture_output=True,
            text=True)

    def test_HIGGS_ENV_is_required_by_name(self):
        """Its default was `$HOME/anaconda3/envs/higgs3` - ONE MACHINE'S conda
        layout. A caller that forgot it got a server out of a directory nobody
        named, or `vllm-omni: No such file` at the end of a launch."""
        done = self.run_script(HIGGS_ENV=None)
        self.assertEqual(done.returncode, 5, done.stderr)
        self.assertIn('HIGGS_ENV is not set', done.stderr)
        self.assertNotIn('anaconda3', done.stderr)

    def test_an_unset_deploy_config_means_the_certified_sibling(self):
        """vllm-omni's own auto-discovered profile caps stage 0 at 2048 frames
        (81.92 s) and no request parameter can raise it, so silence used to mean
        "truncate every long chunk"."""
        done = self.run_script(HIGGS_ENV=self.prefix)
        self.assertEqual(done.returncode, 0, done.stderr)
        expected = os.path.join(os.path.dirname(self.script),
                                v3_served.PACKAGED_DEPLOY_CONFIG)
        printed = [line for line in done.stdout.splitlines()
                   if line.startswith('DEPLOY_CONFIG=')]
        self.assertEqual(len(printed), 1, done.stdout)
        self.assertTrue(printed[0].endswith(v3_served.PACKAGED_DEPLOY_CONFIG),
                        printed[0])
        self.assertTrue(os.path.isfile(expected), expected)
        self.assertIn('--deploy-config', done.stdout)

    def test_an_explicitly_empty_deploy_config_means_vllm_omnis_own(self):
        """`${VAR-...}` and not `${VAR:-...}`: unset is "nobody said", empty is
        a decision. BookForge's catalog needs the distinction - `deployConfig:
        null` there is a choice and a missing key is refused."""
        done = self.run_script(HIGGS_ENV=self.prefix, HIGGS_DEPLOY_CONFIG='')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('DEPLOY_CONFIG=(vllm-omni default profile)', done.stdout)
        self.assertNotIn('--deploy-config', done.stdout)

    def test_it_still_refuses_the_other_stack(self):
        """The half of the HIGGS_STACK contract that lives in the launcher: a
        job configured for SGLang-Omni that reached this script would come up on
        a server whose requests the client is not building."""
        done = self.run_script(HIGGS_ENV=self.prefix, HIGGS_STACK='sglang-omni')
        self.assertEqual(done.returncode, 6, done.stdout)
        self.assertIn('serve_higgs_sgl.sh', done.stderr)


if __name__ == '__main__':
    unittest.main()
