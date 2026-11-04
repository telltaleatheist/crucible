"""What EVERY Higgs v3 serving stack shares: ownership, teardown, and the guest.

TWO STACKS SERVE THE SAME MODEL, and narrator now has both:

  ``v3_served.py``    vllm-omni 0.28.0, conda env ``higgs3``, port 8095.
  ``sgl_served.py``   SGLang-Omni 0.1.4, conda env ``sglomni``, port 8200.

They differ in the launch line, the request body, the frame-cap field and the
context window - and they do NOT differ in anything below. A server is a process
in a WSL guest that narrator started, holds ~19-24 GB of one card, and must come
down when the job does. That problem was solved once (2026-09-05, for
vllm-omni) and every part of the solution is stack-neutral:

  * THE OWNERSHIP MARKER. ``NARRATOR_HIGGS3_OWNER`` is exported into the launch
    wrapper's environment and is therefore in ``/proc/<pid>/environ`` of the
    server and every process it forks. A server is OURS iff the process
    listening on our port carries it. This replaced a recorded pid, which cannot
    follow a process that re-sessions itself after exec (measured: vllm-omni
    84072 -> 84096 -> 84098, so the wrapper's ``$!`` was dead within a second and
    a 24 GB server outlived the job).
    ONE MARKER FOR BOTH STACKS, deliberately: the question it answers is "did
    narrator start this", not "which library is it", and the two stacks bind
    different ports anyway. A leftover from EITHER stack is therefore
    recognisable as ours and can be reclaimed cooperatively rather than refused
    as a stranger's.
  * THE LISTENER SCAN. ``/proc/net/tcp{,6}`` plus every process's fd table - no
    ``ss``, no ``lsof``, both of which are optional packages in a distro.
  * THE WATCHDOG. A detached stdlib python3 in the guest that watches the OWNER
    pid and SIGTERMs the marked listener's process GROUP when it is gone. That
    is what makes "hit Stop" and "the app died" bring the server down; a worker
    killed with SIGKILL never runs its own cleanup.
  * TERM AND ONLY TERM. A SIGKILL to a process holding the GPU inside WSL wedges
    the whole VM until a Windows reboot (memory: wsl-wedge-proofing). There is
    no KILL anywhere in this file and ``_signal_guest`` refuses one BY NAME.

WHAT IS **NOT** SHARED, and lives in the stack modules: the launch command, the
request body, sampling placement (vllm-omni: ``extra_params``; SGLang: the
request's TOP LEVEL), the frame-cap field name, the context window, the model
identity source, and the sentinel-filter patch - which is vllm-omni's alone
(SGLang-Omni has its own stage processor and needs no patch).
"""
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from ..log import log


class HiggsServerError(RuntimeError):
    """The server refused a request, or would not come up.

    Always carries the server's own message: these stacks' 4xx/5xx bodies say
    exactly what is wrong ("Reference audio too long (42.0s)", "Token id -100 is
    out of vocabulary" when vllm-omni's patch is missing) and paraphrasing them
    loses the one thing a reader needs.
    """


class HiggsServerDown(HiggsServerError):
    """Nothing answered at all: the server is not there.

    Distinct from a refusal so a BATCH can tell the two apart. A 400 on one
    chunk is that chunk's failure and the rest of the batch proceeds; a
    connection refused is the whole server gone, and rendering the remaining
    thousand chunks against it would mark every one of them failed one at a
    time.
    """


#: THE OWNERSHIP MARKER. See the module docstring. Its value is the pid of the
#: narrator process that launched the server, prefixed ``win32:`` on the Windows
#: arm where that pid is a HOST pid the guest cannot watch.
OWNER_ENV = 'NARRATOR_HIGGS3_OWNER'

#: WHICH MERGED CHECKPOINT AN ATTACHED SERVER IS RUNNING, asserted by the
#: operator. Consulted only when the running server cannot be made to say so
#: itself, and never allowed to override something the server (or its own
#: environ) reports - a reported path is a fact and an env var is a claim.
CHECKPOINT_ENV = 'NARRATOR_HIGGS3_CHECKPOINT'

#: The launch script's own model-directory knob, exported into the wrapper by a
#: launching backend and read back OUT OF THE SERVER'S ENVIRON by the listener
#: scan. Both stacks' launchers read it.
SERVE_MODEL_DIR_ENV = 'HIGGS_MODEL_DIR'

#: THE ONE EXTERNAL NAME FOR THE CONTEXT WINDOW, across all three Higgs
#: backends. Crucible sets it per voice in the spawn environment, beside
#: `HIGGS_MAX_NUM_SEQS` and `HIGGS_SGL_MEM_FRACTION` (a voice change is a full
#: engine restart, so per-voice IS per-spawn); narrator translates it into
#: whatever each stack actually takes, and each stack's own default applies
#: when it is unset.
#:
#:   sglang-omni  `HiggsTtsEngineBuilder.context_length`, a CLASS ATTRIBUTE
#:                with no flag (engine_builder.py:29, `context_length = 4096`).
#:                narrator's launch entry assigns it in the engine process
#:                before the engine is built - see `launch/sgl_omni_entry.py`.
#:   vllm-omni    `--max-model-len`, which the launcher reads as
#:                `HIGGS_MAX_MODEL_LEN`. narrator exports that from this name;
#:                both set and disagreeing is refused by name.
#:   mlx          `HiggsV3MlxConfig.context_tokens`, wired at load.
CONTEXT_LENGTH_ENV = 'HIGGS_CONTEXT_LENGTH'


def context_length(default: int) -> int:
    """`HIGGS_CONTEXT_LENGTH`, or `default` - THIS STACK's own window.

    `default` is required and is the number the stack uses when nobody said
    otherwise (4096 on sglang-omni, 8192 on vllm-omni and MLX). It is not a
    fallback for a missing required value: an unset variable means "serve the
    window this stack has always served", which is a different statement from
    "narrator could not find out".

    REFUSED, NEVER COERCED: a value that is not a positive whole number raises
    here, before a server is launched or a chunk is sized against it. A context
    that silently became 0 or 4096.0 would size every `max_new_tokens` in the
    book off a number nobody chose.

    NOT CHECKED AGAINST THE MODEL'S OWN MAXIMUM, and this is a stated gap.
    narrator never loads the weights on either served stack - it hands a server
    a path and talks to it over HTTP - so `max_position_embeddings` is not a
    number it has. The MLX arm does load the model, but through mlx-audio's own
    loader, and the Higgs v3 config it exposes carries no position ceiling this
    code has verified. So a value above what the weights support is passed
    through, and what refuses it is the server's own startup (sglang) or the
    first request that runs past the rotary table. Closing this means reading
    the model directory's `config.json` here, which is a change with its own
    measurement to take.
    """
    default = int(default)
    raw = (os.environ.get(CONTEXT_LENGTH_ENV) or '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f'{CONTEXT_LENGTH_ENV}={raw!r} is not a whole number of tokens. It '
            'is the model context every request is sized against; there is '
            'nothing to round it to.') from None
    if value < 1:
        raise ValueError(
            f'{CONTEXT_LENGTH_ENV}={raw!r} must be at least 1 token.')
    return value

#: How long one guest-side command (a /proc scan, a group signal) may take
#: before it is reported as unanswered. Owen, 2026-09-05: "timeouts are intended
#: to kill something if its waiting for an obscenely long time ... it should be
#: like 10 minutes." These are wedge detectors, not budgets: a WSL VM that takes
#: a minute to answer under load is slow, not gone, and a 30 s ceiling turned
#: slow into "could not scan" on a healthy machine.
GUEST_COMMAND_TIMEOUT_SECONDS = 10 * 60

#: How long the guest-side watchdog sleeps between looks at the owner.
WATCHDOG_INTERVAL_SECONDS = 3

#: How often `wait_ready` asks whether the server it launched went to a port
#: nobody asked for. Not every second: the scan is a process spawn (`wsl.exe` on
#: the Windows arm) and a misbind is a permanent state, so answering it half a
#: minute late costs half a minute out of a fifteen-minute patience and saves
#: the other fourteen and a half.
MISBIND_CHECK_INTERVAL_SECONDS = 30.0

#: WHICH SERVING STACK THIS JOB RENDERS ON. BookForge sets it from the catalog's
#: `serving.stack`, on every arm and every phase, exactly as it sets
#: HIGGS_MAX_NUM_SEQS - and for the same reason: the two stacks want different
#: request bodies, different frame caps and different sampling, so a job that
#: does not say which one it is on cannot be rendered correctly by guessing.
STACK_ENV = 'HIGGS_STACK'

#: vllm-omni 0.28.0 (env `higgs3`, port 8095). The original arm.
STACK_VLLM_OMNI = 'vllm-omni'
#: SGLang-Omni 0.1.4 (env `sglomni`, port 8200). 2.5x the throughput and clean
#: at 16 in flight - see `sgl_served`'s docstring for the measurements.
STACK_SGLANG_OMNI = 'sglang-omni'

STACKS = (STACK_VLLM_OMNI, STACK_SGLANG_OMNI)


def serving_stack() -> str:
    """Which serving stack this process renders on - `HIGGS_STACK`.

    REFUSED BY NAME WHEN UNSET. There is no default, and the reason is that the
    two stacks are not interchangeable in any of the places it matters:
    vllm-omni reads a checkpoint's `generation_config.json` for itself and takes
    per-request sampling in `extra_params`, while SGLang-Omni reads no such file
    and applies NO top_k/top_p unless the request carries them at the top level
    (measured: without them a chunk ran to the cap with 80 s of silence). A
    guessed stack is therefore not a smaller failure than a crash - it is a book
    rendered at sampling nobody chose, or a request whose fields were dropped in
    silence.

    The same contract as `serve_concurrency`'s `HIGGS_MAX_NUM_SEQS`: BookForge
    states it on every door from the catalog's `serving.stack`.
    """
    raw = (os.environ.get(STACK_ENV) or '').strip()
    if not raw:
        raise ValueError(
            f'Higgs: {STACK_ENV} is not set. It names the serving stack this job '
            f"renders on - one of {', '.join(STACKS)} - and BookForge sets it "
            "from the catalog's serving.stack on every phase. There is no "
            'default: the two stacks place sampling differently (vllm-omni in '
            'extra_params, SGLang-Omni at the request top level), size the frame '
            'cap against different context windows (8192 vs a hard-coded 4096), '
            'and disagree about whether an empty sampling means "the '
            'checkpoint\'s own numbers" or "the untruncated codebook tail".')
    if raw not in STACKS:
        raise ValueError(
            f'Higgs: {STACK_ENV}={raw!r} is not a serving stack narrator has. '
            f"The stacks are {', '.join(STACKS)}.")
    return raw


def to_wsl(path: str) -> str:
    r"""A Windows path -> the path WSL sees.

        C:\x\y                    -> /mnt/c/x/y
        \\wsl$\Ubuntu\home\t        -> /home/t   (already INSIDE the distro)
        \\wsl.localhost\Ubuntu\opt  -> /opt
        /already/posix             -> unchanged

    The UNC forms matter and are not theoretical: a script living in the
    distro's own filesystem is reached from Windows as `\\wsl$\<distro>\...`,
    and running that through the drive-letter rule would produce a path with no
    meaning on either side - the launch would fail with a confusing "No such
    file" from bash rather than from here.
    """
    path = (path or '').replace('\\', '/')
    lowered = path.lower()
    for prefix in ('//wsl$/', '//wsl.localhost/'):
        if lowered.startswith(prefix):
            # Drop the prefix AND the distro name; what remains is an absolute
            # path in the guest.
            _distro, _, tail = path[len(prefix):].partition('/')
            return '/' + tail
    if len(path) > 1 and path[1] == ':':
        return '/mnt/' + path[0].lower() + path[2:]
    return path


class GuestOwnedServer:
    """A model server this process launched inside a WSL guest, and owns.

    Subclasses supply the STACK: `launch_command()`, `check_serves_expected_model()`
    and whatever the request looks like. This class owns the lifecycle -
    ownership, health, adoption, teardown - and nothing about either library.

    Two modes, chosen by what a subclass's `__init__` was given:

      ATTACH   `base_url` names a server somebody else started. `start()` is a
               no-op and `stop()` refuses to kill a process it did not launch.
      LAUNCH   `serve_script` names the launcher. `start()` runs it - through
               `wsl.exe -d <distro> --exec bash -c` on Windows, directly on
               Linux - and `stop()` terminates its process GROUP.
    """

    #: The health endpoint both stacks expose. Overridable, stated rather than
    #: assumed, because "something answers this port" is the weakest possible
    #: identity claim and every subclass follows it with a real one.
    HEALTH_PATH = '/health'

    #: What this backend's log lines are prefixed with, so a run log says which
    #: stack spoke.
    LOG_TAG = '[HIGGS]'

    #: Named in `verify_gone`/`proof_log` refusals: where an operator states an
    #: ATTACHED server's log. Subclasses set their own.
    SERVER_LOG_ENV = 'NARRATOR_HIGGS3_SERVER_LOG'

    #: What `wait_ready` says a dead launch is usually caused by. Stack-specific
    #: because the two stacks fail to start for different reasons.
    READY_FAILURE_HINT = ''

    # Set by subclass __init__ before any method here is called.
    base_url = ''
    serve_script = None
    wsl_distro = 'Ubuntu'
    launch_log = ''
    server_log = None
    spec = None
    _named_log = None
    _log_is_ours = False
    _log_handle = None
    _proc = None
    _guest_pid = None

    # -- ownership -----------------------------------------------------------

    def owner_id(self) -> str:
        """The value the server's environment carries in OWNER_ENV: this
        process's pid. On the Windows arm it is a HOST pid the guest cannot
        watch, and it is prefixed to say so, so the watchdog knows not to."""
        pid = os.getpid()
        return f'win32:{pid}' if sys.platform == 'win32' else str(pid)

    #: Shared by the watchdog and the ownership scan: the pids LISTENING on a
    #: TCP port, from /proc/net/tcp{,6} and every process's fd table. No `ss`,
    #: no `lsof` - both are optional packages in a distro; /proc is not.
    #:
    #: `env_of` reads ANY variable out of a process's environ, which is what
    #: makes the SERVER'S OWN LAUNCH ENVIRONMENT readable: `HIGGS_MODEL_DIR` in
    #: there is a fact about the running process, not a claim by whoever is
    #: asking. SGLang-Omni's `/v1/models` reports its SERVED NAME as `root`
    #: rather than the model path (read off sglang_omni/serve/openai_api.py's
    #: `_register_models`: `ModelCard(id=model_name, root=model_name)`), so on
    #: that stack this is where "which checkpoint is up" is answered.
    _LISTENERS_PY = r"""
import os
def listeners(port):
    inodes = set()
    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            rows = open(table).read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            f = row.split()
            if len(f) < 10 or f[3] != '0A':
                continue
            if int(f[1].rsplit(':', 1)[1], 16) == port:
                inodes.add(f[9])
    pids = []
    if not inodes:
        return pids
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            fds = os.listdir('/proc/%s/fd' % entry)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink('/proc/%s/fd/%s' % (entry, fd))
            except OSError:
                continue
            if target.startswith('socket:[') and target[8:-1] in inodes:
                pids.append(int(entry))
                break
    return pids
def env_of(pid, name):
    key = (name + '=').encode('ascii')
    try:
        env = open('/proc/%d/environ' % pid, 'rb').read().split(b'\0')
    except OSError:
        return None
    for item in env:
        if item.startswith(key):
            return item.split(b'=', 1)[1].decode('utf-8', 'replace')
    return None
def owner_of(pid):
    return env_of(pid, 'NARRATOR_HIGGS3_OWNER')
def pgid_of(pid):
    stat = open('/proc/%d/stat' % pid).read()
    return int(stat[stat.rindex(')') + 2:].split()[2])
"""

    #: `_misbound_ports`'s program: every TCP port a process carrying OUR marker
    #: is LISTENING on, as `[{'pid', 'port'}]`.
    #:
    #: THE INVERSE QUESTION to `_OWN_SERVERS_SCAN`'s. That one asks "who is on
    #: the port I asked for"; this asks "where did the thing I launched actually
    #: go". See `_refuse_if_misbound` for the measurement that made the second
    #: question necessary.
    _BOUND_PORTS_SCAN = _LISTENERS_PY + r"""
import json, sys
mark = sys.argv[1]
inode_port = {}
for table in ('/proc/net/tcp', '/proc/net/tcp6'):
    try:
        rows = open(table).read().splitlines()[1:]
    except OSError:
        continue
    for row in rows:
        f = row.split()
        if len(f) < 10 or f[3] != '0A':
            continue
        inode_port[f[9]] = int(f[1].rsplit(':', 1)[1], 16)
found = []
for entry in os.listdir('/proc'):
    if not entry.isdigit():
        continue
    pid = int(entry)
    if owner_of(pid) != mark:
        continue
    try:
        fds = os.listdir('/proc/%s/fd' % entry)
    except OSError:
        continue
    ports = set()
    for fd in fds:
        try:
            target = os.readlink('/proc/%s/fd/%s' % (entry, fd))
        except OSError:
            continue
        if target.startswith('socket:['):
            port = inode_port.get(target[8:-1])
            if port is not None:
                ports.add(port)
    for port in sorted(ports):
        found.append({'pid': pid, 'port': port})
print(json.dumps(found))
"""

    #: `_own_servers_on_port`'s program: one JSON list of the marked listeners,
    #: each with the MODEL DIRECTORY its own environ carries (None when the
    #: launcher exported none - the base weights).
    _OWN_SERVERS_SCAN = _LISTENERS_PY + r"""
import json, sys
port = int(sys.argv[1])
found = []
for pid in listeners(port):
    owner = owner_of(pid)
    if owner is None:
        continue
    try:
        pgid = pgid_of(pid)
    except (OSError, ValueError):
        continue
    found.append({'pid': pid, 'pgid': pgid, 'owner': owner,
                  'modelDir': env_of(pid, 'HIGGS_MODEL_DIR')})
print(json.dumps(found))
"""

    #: The watchdog's program. See the module docstring, item 3.
    _WATCHDOG = _LISTENERS_PY + r"""
import signal, sys, time
owner = int(sys.argv[1]); port = int(sys.argv[2]); every = float(sys.argv[3])
mark = str(owner)
def ours():
    return [p for p in listeners(port) if owner_of(p) == mark]
started = time.time()
seen = False
while True:
    time.sleep(every)
    if os.path.exists('/proc/%d' % owner):
        # The owner is alive. Once the server it launched has been seen and is
        # gone (an ordinary stop), there is nothing left to guard: exit, so the
        # wrapper's `wait` - which waits for this process too - returns with the
        # server instead of holding stop() for its whole timeout (every unload
        # took ~60 s on the PC, 2026-10-02).
        if ours():
            seen = True
        elif seen:
            break
        continue
    # The owner is gone. Take down what it launched, by group, TERM only.
    groups = set()
    for p in ours():
        try:
            groups.add(pgid_of(p))
        except (OSError, ValueError):
            pass
    for g in groups:
        try:
            os.killpg(g, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + 180
    while time.time() < deadline and ours():
        time.sleep(1)
    break
"""

    #: `_signal_guest`'s program: SIGTERM the process GROUP of one pid, the
    #: group read at this moment. Prints the pgid it signalled.
    _SIGNAL_GROUP = r"""
import os, signal, sys
pid = int(sys.argv[1])
stat = open('/proc/%d/stat' % pid).read()
pgid = int(stat[stat.rindex(')') + 2:].split()[2])
os.killpg(pgid, signal.SIGTERM)
print(pgid)
"""

    def _watchdog_clause(self) -> str:
        """The wrapper fragment that starts the guest-side watchdog, or ''.

        NOT STARTED ON THE WINDOWS ARM: the owner is a host pid the guest cannot
        see (`owner_id` says so in the marker's own prefix), so there is nothing
        for the watchdog to watch and `stop()` is the only teardown.
        """
        if sys.platform == 'win32':
            return ''
        import shlex
        port = self.base_url.rsplit(':', 1)[-1].rstrip('/')
        return (f'setsid python3 -c {shlex.quote(self._WATCHDOG)} '
                f'{os.getpid()} {shlex.quote(port)} {WATCHDOG_INTERVAL_SECONDS} '
                f'>/dev/null 2>&1 < /dev/null & ')

    def _guest_argv(self, argv: list) -> list:
        """`argv`, run INSIDE the distro on Windows and directly elsewhere.

        `--exec` on the Windows arm because without it wsl.exe hands the line to
        the distro's default shell, which expands `$` before the program sees
        it (memory: wsl-exe-implicit-shell-trap). `--` is NOT a substitute.
        """
        if sys.platform != 'win32':
            return list(argv)
        wsl = shutil.which('wsl.exe') or 'wsl.exe'
        return [wsl, '-d', self.wsl_distro, '--exec'] + list(argv)

    def _own_servers_on_port(self) -> list:
        """Every server NARRATOR started that is listening on our port, as
        `[{'pid', 'pgid', 'owner', 'modelDir'}]`: the listeners whose environment
        carries OWNER_ENV. A listener without it is somebody else's. Raises if
        the scan itself cannot run - an unanswerable ownership question is not a
        "no"."""
        port = self.base_url.rsplit(':', 1)[-1].rstrip('/')
        argv = self._guest_argv(['python3', '-c', self._OWN_SERVERS_SCAN, port])
        try:
            out = subprocess.run(argv, capture_output=True, text=True,
                                 timeout=GUEST_COMMAND_TIMEOUT_SECONDS)
        except (OSError, subprocess.SubprocessError) as exc:
            raise HiggsServerError(
                f'Higgs: could not scan for narrator\'s own servers on port '
                f'{port} ({exc}); refusing to decide whether the server already '
                'there is ours.') from exc
        if out.returncode != 0:
            raise HiggsServerError(
                f'Higgs: the ownership scan for port {port} failed (exit '
                f'{out.returncode}): {out.stderr.strip()[:400]}')
        try:
            return json.loads(out.stdout.strip() or '[]')
        except ValueError as exc:
            raise HiggsServerError(
                f'Higgs: the ownership scan for port {port} printed '
                f'{out.stdout[:200]!r}, not JSON.') from exc

    def _bound_ports(self) -> list:
        """Every `{'pid', 'port'}` a process carrying OUR marker is listening on.

        Raises if the scan cannot run, for the same reason
        `_own_servers_on_port` does: an unanswerable question about our own
        process is not a "no".
        """
        argv = self._guest_argv(['python3', '-c', self._BOUND_PORTS_SCAN,
                                 self.owner_id()])
        try:
            out = subprocess.run(argv, capture_output=True, text=True,
                                 timeout=GUEST_COMMAND_TIMEOUT_SECONDS)
        except (OSError, subprocess.SubprocessError) as exc:
            raise HiggsServerError(
                f'Higgs: could not scan for the ports our own server bound '
                f'({exc}); refusing to decide whether it came up on the port it '
                'was told to use.') from exc
        if out.returncode != 0:
            raise HiggsServerError(
                f'Higgs: the bound-port scan failed (exit {out.returncode}): '
                f'{out.stderr.strip()[:400]}')
        try:
            return json.loads(out.stdout.strip() or '[]')
        except ValueError as exc:
            raise HiggsServerError(
                f'Higgs: the bound-port scan printed {out.stdout[:200]!r}, '
                'not JSON.') from exc

    def _answers_health(self, port: int) -> bool:
        """True when something on `port` answers OUR health endpoint with 200."""
        host = self.base_url.split('://', 1)[-1].rstrip('/').rpartition(':')[0]
        try:
            with urllib.request.urlopen(f'http://{host}:{port}{self.HEALTH_PATH}',
                                        timeout=3) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _refuse_if_misbound(self) -> None:
        """Refuse, BY NAME, when our server came up on a port nobody asked for.

        MEASURED: `sgl-omni` does not fail when the port it is given is busy. It
        SILENTLY BINDS A RANDOM ONE - told 8200, it took 57877, and on the next
        start 34529 - while narrator polled 8200. `/health` on 8200 never
        answers, the process never dies, so `wait_ready` sat out its entire
        patience (fifteen minutes on the v3 default) waiting for a condition
        that had already been made impossible, and then reported a timeout that
        pointed at cold starts and flashinfer. Nothing in that sequence was
        wrong except the question being asked.

        SO "THE PORT IS LISTENING" IS NECESSARY AND NOT SUFFICIENT: it has to be
        THE PORT WE ASKED FOR, and a mismatch is a failure with a name rather
        than more waiting. There is nothing to wait for - the server is up, the
        render will never reach it, and a second launch onto the same busy port
        would do the same thing again.

        TWO FACTS ARE REQUIRED BEFORE REFUSING, because one is not enough. A
        process carrying our marker listening on some other port is ORDINARY:
        vLLM's engine-core and SGLang's schedulers are separate processes that
        inherit our environment and open TCP sockets of their own, so the marker
        alone would refuse every healthy launch. The second fact is that the
        stranger port ANSWERS OUR HEALTH ENDPOINT with 200 - which an internal
        IPC socket does not, and a misbound server does.

        Only ever called while LAUNCHING: in attach mode the operator named the
        URL and there is no "port we asked for" to be missed.
        """
        if not self.serve_script:
            return
        asked = int(self.base_url.rsplit(':', 1)[-1].rstrip('/'))
        for row in self._bound_ports():
            port = int(row['port'])
            if port == asked or not self._answers_health(port):
                continue
            # TAKE IT DOWN BEFORE REFUSING. It carries OUR marker (this is what
            # `_bound_ports` scans for), so it is ours, and raising first left it
            # orphaned: `_record_server` had never run, `stop()` had no pid to
            # signal, and a setsid'd sgl-omni sat on 13.4 GB of VRAM until the
            # next job came back accelerator_busy (Crucible 1.0.43, my-pc,
            # 2026-09-27).
            self._stop_misbound(int(row['pid']), port)
            raise HiggsServerError(
                f'Higgs: the server narrator launched (pid {row["pid"]}) is '
                f'answering {self.HEALTH_PATH} on port {port}, NOT on port '
                f'{asked}, which is the port it was told to bind and the port '
                'every render would be sent to. This is what sgl-omni does when '
                'the port it is given is already busy: it takes a random free '
                'one and says nothing (measured on example-pc - told 8200, bound '
                '57877, then 34529). Waiting is pointless, because the server is '
                'already up and can never appear on the port that was asked for. '
                f'Free port {asked} - something else is on it - and start again.')

    def _stop_misbound(self, pid: int, port: int, timeout: float = 180.0) -> None:
        """SIGTERM the group of our own server that came up on `port` rather
        than the port it was asked for, and wait for that port to stop answering.

        `_verify_gone` cannot do the waiting: it polls OUR port, where nothing
        ever answered. TERM only, never KILL, for `_verify_gone`'s reason: a KILL
        to a GPU holder inside WSL wedges the VM. A survivor is reported by pid,
        and it keeps our marker, so the watchdog and the next start can still
        recognise it."""
        log(f'{self.LOG_TAG} our server (pid {pid}) came up on port {port}, not '
            'the port it was asked for; stopping it before refusing', flush=True)
        self._signal_guest(pid, 'TERM')
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if not self._answers_health(port):
                return
            time.sleep(1.0)
        log(f'{self.LOG_TAG} WARNING: our misbound server (pid {pid}) is still '
            f'answering on port {port} {timeout:.0f}s after SIGTERM. NOT killing '
            'it - a KILL on a GPU holder inside WSL wedges the VM.', flush=True)

    def _server_on_port(self):
        """The server listening on our port that carries OUR marker (any
        narrator's - the owner value is not compared here), or None. Raises if
        the scan cannot run: an unanswerable ownership question is not a "no"."""
        rows = self._own_servers_on_port()
        return rows[0] if rows else None

    def _record_server(self) -> None:
        """After `/health` answers: find the listener with our marker and
        remember its pid. Its GROUP is read again at signal time, never
        remembered - a group id is a property of the moment."""
        row = self._server_on_port()
        if row is None:
            log(f'{self.LOG_TAG} WARNING: {self.base_url} answers but no listener '
                f'on its port carries {OWNER_ENV}; stop() will have nothing of '
                'ours to signal.', flush=True)
            self._guest_pid = None
            return
        self._guest_pid = int(row['pid'])
        log(f'{self.LOG_TAG} server pid {row["pid"]} (group {row["pgid"]}, owner '
            f'{row["owner"]})', flush=True)

    def _record_unhealthy_own_server(self) -> None:
        """`_record_server` for a server that never answered health: the
        listener on our port whose marker is THIS process's owner id. A scan
        that cannot run is logged, not raised - this is a teardown, and the
        watchdog still stands behind it."""
        try:
            rows = self._own_servers_on_port()
        except HiggsServerError as exc:
            log(f'{self.LOG_TAG} WARNING: could not look for our own unhealthy '
                f'server on {self.base_url} ({exc}); the watchdog is what stops '
                'it now.', flush=True)
            return
        mine = [r for r in rows if r.get('owner') == self.owner_id()]
        if not mine:
            return
        self._guest_pid = int(mine[0]['pid'])
        log(f'{self.LOG_TAG} our server (pid {mine[0]["pid"]}) is on '
            f'{self.base_url} but never became healthy; stopping it', flush=True)

    def _signal_guest(self, pid: int, signame: str) -> None:
        """SIGTERM the process GROUP of `pid`, inside the distro, the group read
        off /proc at this moment - never a remembered one, and never a pattern
        (`pkill -f "vllm-omni serve"` would kill another agent's server too).
        `KILL` is refused by name: see `_verify_gone`."""
        if signame != 'TERM':
            raise ValueError(
                f'Higgs: refusing to send SIG{signame} to a server process. '
                'A KILL on a process holding the GPU inside WSL wedges the VM; '
                'only TERM is sent, and a server that ignores it is reported, '
                'not killed.')
        argv = self._guest_argv(['python3', '-c', self._SIGNAL_GROUP, str(int(pid))])
        try:
            out = subprocess.run(argv, capture_output=True, text=True,
                                 timeout=GUEST_COMMAND_TIMEOUT_SECONDS)
        except (OSError, subprocess.SubprocessError) as exc:
            log(f'{self.LOG_TAG} could not signal the group of pid {pid}: {exc}',
                flush=True)
            return
        if out.returncode != 0:
            log(f'{self.LOG_TAG} could not signal the group of pid {pid}: '
                f'{out.stderr.strip()[:300]}', flush=True)
        else:
            log(f'{self.LOG_TAG} SIGTERM sent to process group '
                f'{out.stdout.strip()} (pid {pid})', flush=True)

    def _reclaim_port(self, wrong: HiggsServerError, timeout: float = 180.0) -> None:
        """The server on our port is the wrong one. Take it down IF IT IS OURS,
        cooperatively, and wait for the port to free; otherwise re-raise the
        refusal, now saying whose it is not.

        Cooperative means SIGTERM to the process group and patience: a server
        mid-teardown holds CUDA state, and a SIGKILL to a process holding the GPU
        inside WSL wedges the whole VM until a Windows reboot. So there is no
        KILL here, only a longer wait and then a refusal that names the pid.
        """
        owned = self._own_servers_on_port()
        if not owned:
            raise HiggsServerError(
                f'{wrong} The listener on that port carries no {OWNER_ENV}, so it '
                'is not a server narrator started and it will not be stopped '
                'from here. Stop it yourself, or attach to a server running the '
                'right checkpoint.') from wrong
        for row in owned:
            log(f'{self.LOG_TAG} the server on {self.base_url} is the wrong '
                f'checkpoint and it is narrator\'s (pid {row["pid"]}, group '
                f'{row["pgid"]}, launched by {row["owner"]}); stopping it before '
                'launching', flush=True)
            self._signal_guest(row['pid'], 'TERM')
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if not self.ping():
                break
            time.sleep(1.0)
        else:
            raise HiggsServerError(
                f'Higgs: narrator\'s own server on {self.base_url} (pid(s) '
                f'{", ".join(str(r["pid"]) for r in owned)}) is still answering '
                f'{timeout:.0f}s after SIGTERM. It is NOT being killed: a KILL to '
                'a process holding the GPU inside WSL wedges the VM. Wait for it, '
                'or stop it yourself, then retry.')
        log(f'{self.LOG_TAG} port {self.base_url.rsplit(":", 1)[-1]} reclaimed',
            flush=True)

    # -- the log -------------------------------------------------------------

    def default_launch_log(self, prefix: str) -> str:
        """A per-INSTANCE temp path for a launch nobody gave a process dir. Per
        instance and not per process: two workers must never share one file."""
        return os.path.join(
            tempfile.gettempdir(), f'{prefix}-{os.getpid()}-{id(self):x}.log')

    def _open_log(self) -> None:
        """Open (and truncate) the launch log. Failure is LOUD.

        A log we cannot open is not a cosmetic loss: it is the only record of
        what the server did, and starting a ~19 GB server whose evidence goes
        nowhere is the state `wait_ready`'s "check its log" used to point at.
        """
        directory = os.path.dirname(self.launch_log)
        try:
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._log_handle = open(self.launch_log, 'wb')
        except OSError as exc:
            raise HiggsServerError(
                f'Higgs: could not open the server log {self.launch_log} '
                f'({exc}). That file is where this server\'s stdout and stderr '
                'go, and a server started without it renders with no evidence of '
                'what its decode path did.') from exc
        log(f'{self.LOG_TAG} server log: {self.launch_log}', flush=True)

    def _close_log(self) -> None:
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def proof_log(self):
        """The log a proof reads, or None.

        Ours when we launched the server; the operator's named file when we
        attached to (or adopted) somebody else's. None when neither - which is an
        honest answer and not a path to guess at.
        """
        return self.launch_log if self._log_is_ours else self._named_log

    # -- lifecycle -----------------------------------------------------------

    def ping(self) -> bool:
        """True when the health endpoint answers 200."""
        try:
            with urllib.request.urlopen(self.base_url + self.HEALTH_PATH,
                                        timeout=3) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def wait_ready(self, timeout: float) -> bool:
        """Poll the health endpoint until it answers, or `timeout` seconds pass.

        Returns False on timeout rather than raising - a slow start is the
        caller's decision, and cold starts are minutes on both stacks. RAISES on
        the two states where waiting can no longer produce an answer:

          * THE PROCESS WE LAUNCHED HAS DIED, naming its exit status. Waiting out
            a timeout on a corpse is the failure mode this originally existed to
            avoid.
          * THE SERVER IS UP ON THE WRONG PORT (`_refuse_if_misbound`). Also a
            state the wait can never leave, and until 2026-09-13 it was
            indistinguishable from a slow cold start - which on the v3 default
            means fifteen minutes of patience spent on something that had already
            happened. Checked on an interval rather than every second because the
            scan is a process spawn (a `wsl.exe` one on the Windows arm) and a
            misbind is answered just as usefully half a minute late.
        """
        deadline = time.time() + float(timeout)
        next_bind_check = time.time() + MISBIND_CHECK_INTERVAL_SECONDS
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise HiggsServerError(
                    f'Higgs server exited with status {self._proc.returncode} '
                    f'before becoming ready. Its log is {self.launch_log}. '
                    + self.READY_FAILURE_HINT)
            if self.ping():
                return True
            if time.time() >= next_bind_check:
                self._refuse_if_misbound()
                next_bind_check = time.time() + MISBIND_CHECK_INTERVAL_SECONDS
            time.sleep(1.0)
        # ONE LAST LOOK BEFORE REPORTING A TIMEOUT. A misbind that happened in
        # the final interval would otherwise be reported as "the server did not
        # come up", which is the message that sent an afternoon after flashinfer
        # and gpu-memory-utilization while sgl-omni sat healthy on port 57877.
        self._refuse_if_misbound()
        return False

    def start(self) -> None:
        """Launch the server. Idempotent: a second call while it is up does
        nothing, and never a second process on the same port.

        ADOPTION IS CONDITIONAL ON IDENTITY. A port is not proof: a leftover
        server from another checkpoint, another stack or another agent's session
        answers health identically and would render a whole book in the wrong
        voice while every message here named the right one. So an already-serving
        port is adopted only if `check_serves_expected_model()` passes, and
        otherwise reclaimed (if it is narrator's) or refused (if it is not).
        """
        if not self.serve_script:
            return                     # attach mode: somebody else owns it
        if self._proc is not None and self._proc.poll() is None:
            return
        if self.ping():
            try:
                self.check_serves_expected_model()
            except HiggsServerError as wrong:
                # Never a second launch onto a busy port - that pays a full cold
                # start and ~19 GB to fail to bind.
                self._reclaim_port(wrong)
            else:
                # ADOPTED, NOT LAUNCHED: that process's output goes wherever its
                # own launcher sent it, so the file we would have written is not
                # the proof stream and the spec must stop claiming it is.
                self._log_is_ours = False
                self.server_log = self._named_log
                if self.spec is not None:
                    self.spec = dataclasses.replace(self.spec,
                                                    server_log=self.server_log)
                log(f'{self.LOG_TAG} adopting the server already on '
                    f'{self.base_url}; its log is '
                    + (self.server_log
                       or f'not named (see {self.SERVER_LOG_ENV})'), flush=True)
                return
        command = self.launch_command()
        log(f'{self.LOG_TAG} launching: {" ".join(command)}', flush=True)
        # BOTH STREAMS INTO ONE FILE, opened 'wb' so each start overwrites.
        # stderr is merged into stdout because these servers log to stderr and
        # the ordering between the two only means anything interleaved.
        self._open_log()
        # `start_new_session` on POSIX: the wrapper shell leads its own session,
        # so a stop can never reach back into the worker's own group. The server
        # itself gets a further group of its own from the wrapper's `setsid`.
        self._proc = subprocess.Popen(command, stdout=self._log_handle,
                                      stderr=subprocess.STDOUT,
                                      start_new_session=(sys.platform != 'win32'))

    def stop(self, timeout: float = 60.0) -> None:
        """Terminate the server we launched, and VERIFY it is gone.

        Idempotent, and it never kills a process it did not start: in ATTACH mode
        it returns at once, leaving the server running and unpolled.

        THE WINDOWS PROBLEM. On Linux the child IS the server (the launcher
        `exec`s it), so SIGTERM reaches it. On Windows the child is `wsl.exe`,
        and terminating it kills the Windows-side relay - the guest process may
        keep running, holding ~19 GB of VRAM, invisible to `proc.poll()`, which
        now reports a tidy exit. So the SERVER is signalled first, BY GROUP,
        found by its marker inside the distro.
        """
        if not self.serve_script:
            # ATTACH MODE: nothing of ours to terminate. Returning immediately is
            # not a shortcut - polling the port here would block on somebody
            # else's healthy server for the whole timeout and report it as a leak.
            return
        proc = self._proc
        self._proc = None
        if proc is None and self._guest_pid is None:
            self._close_log()
            return
        if self._guest_pid is None and self.ping():
            # Launched, but the server was never recorded (a load that failed
            # before health, say). Find it now by its marker rather than leave it
            # running.
            self._record_server()
        elif self._guest_pid is None and proc is not None:
            # Launched and NEVER healthy (a wait_ready timeout): it may still be
            # listening on our port mid-load, holding the card, with nothing to
            # ping. Ours only if its marker names THIS process, since no health
            # answer says whose checkpoint it is.
            self._record_unhealthy_own_server()
        # THE SERVER FIRST, BY GROUP. Signalling the wrapper shell was the orphan
        # bug: bash does not forward SIGTERM to a backgrounded child, so the
        # shell died, `proc.poll()` reported a tidy exit, and the server ran on
        # with the GPU.
        if self._guest_pid is not None:
            self._signal_guest(self._guest_pid, 'TERM')
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # The wrapper outlived the server's own teardown window. It is a
                # shell, not a GPU holder: terminating IT is safe.
                log(f'{self.LOG_TAG} launch wrapper still up after the server was '
                    'signalled; terminating the wrapper', flush=True)
                try:
                    proc.terminate()
                    proc.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        self._verify_gone(timeout=timeout)
        # Closed LAST, so the server's own shutdown lines are in the file. The
        # file stays: it is the run's evidence.
        self._close_log()

    def _verify_gone(self, timeout: float = 60.0) -> None:
        """Poll the port until nothing answers.

        NO KILL, EVER. A SIGKILL to a process holding the GPU inside WSL wedges
        the whole VM until a Windows reboot, so the escalation after a second
        SIGTERM is a loud warning that names the pid - and the server keeps our
        marker, which is what lets the next `start()` recognise the survivor as
        ours and reclaim the port cooperatively instead of refusing it as a
        stranger's.
        """
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if not self.ping():
                self._guest_pid = None
                return
            time.sleep(1.0)
        if self._guest_pid is None:
            log(f'{self.LOG_TAG} WARNING: something is still serving '
                f'{self.base_url} and this process did not record a guest pid for '
                'it; leaving it alone rather than killing a server it may not '
                'own.', flush=True)
            return
        log(f'{self.LOG_TAG} server still up after the launcher exited; '
            f'signalling guest process group {self._guest_pid} again', flush=True)
        self._signal_guest(self._guest_pid, 'TERM')
        deadline = time.time() + 120.0
        while time.time() < deadline:
            if not self.ping():
                self._guest_pid = None
                return
            time.sleep(1.0)
        log(f'{self.LOG_TAG} WARNING: server pid {self._guest_pid} is still '
            f'serving {self.base_url} after two SIGTERMs. NOT killing it - a KILL '
            'on a GPU holder inside WSL wedges the VM. It still carries our '
            'marker, so the next start reclaims it.', flush=True)

    # -- what a stack must answer --------------------------------------------

    def launch_command(self) -> list:
        raise NotImplementedError

    def check_serves_expected_model(self, checkpoint_dir: str = None) -> None:
        raise NotImplementedError
