/* Crucible operator console — PHASE13-OPERATOR.md section 4.
 *
 * Vanilla, one file, no build step and no dependency. Six sections, each drawn
 * from the read that owns it:
 *
 *   Status      GET /v1/setup      + GET /v1/activity  (+ the card, from /v1/info)
 *   Tasks       GET /v1/tasks      + the running one's SSE
 *   Job types   GET /v1/capability + what /v1/info says this server offers
 *   Catalog     GET /v1/catalog
 *   Connect     GET /v1/setup
 *   Service     GET /v1/info       + GET /v1/setup
 *
 * WHICH READ SAYS A JOB TYPE IS INSTALLED, AND WHY IT IS NOT `/v1/setup`.
 * `setup.job_types` and `info.job_types` are the same list from the same
 * producer — what this server accepts in `POST /v1/jobs` — and that list is
 * spelled in POSTable types: `load-model`, `unload-model`, `tts`, `echo`. The
 * Job types section and the catalog's rows both speak in CAPABILITIES —
 * `llm`, `tts`, `asr` — which is the other of the two lists `/v1/info`
 * deliberately keeps apart, and `info.capabilities[].job_type` is it. Asking
 * `setup.job_types` whether `llm` is here answers no on a server that is
 * serving it, which is the one wrong answer that matters.
 *
 * TWO RULES THE WHOLE FILE IS WRITTEN TO.
 *
 * 1. Every refusal this API names is shown with its CODE and its MESSAGE,
 *    verbatim, beside the control that caused it. There is no toast, no
 *    "something went wrong", and nothing is ever told "maybe"
 *    (ARCHITECTURE.md R3). A 409 that carries a holder shows the holder's own
 *    sentence — `details.who` — because "held by foundry — translate,
 *    qwen3.8-27b-4bit" is the difference between a system working and a button
 *    somebody concludes is broken.
 *
 * 2. Nothing here is a second copy of a server-side table. The job types, the
 *    narrator engines, the subject kinds and what installs what all arrive on
 *    the wire; this file has no list of them and must never grow one.
 *
 * THE EVENTS ARE READ WITH `fetch`, NOT `EventSource`. Every /v1 route needs a
 * bearer token and an API version header, and `EventSource` can send neither.
 * The frame parsing below is the same SSE envelope the job stream uses.
 */

'use strict';

(function () {
  var API_HEADER = 'X-Crucible-Api';
  var API_VERSION = '1';
  var TOKEN_KEY = 'crucible.token';
  var ACTIVITY_MS = 4000;
  var STATUS_THROTTLE_MS = 1500;
  var INSTALL_LINES_KEPT = 400;
  var TERMINAL = ['done', 'failed', 'cancelled'];

  // ------------------------------------------------------------------ state

  var state = {
    token: null,
    setup: null,
    info: null,
    activity: null,
    capability: null,
    catalog: null,
    tasks: [],
    pairingRequests: [],
    refusals: {},
    running: null,
    live: null,
    stream: null,
    revealToken: false,
    // What the OPERATOR has typed or chosen, kept out of the DOM on purpose.
    // Status is on a timer and the whole console is redrawn from it, so a
    // half-typed module or a picked narrator engine that lived only in an
    // element would be thrown away every four seconds.
    moduleText: '',
    engineChoice: {},
    // The settings document, as the server last handed it over — never edited
    // in place. Every control PUTs and replaces this whole object with what
    // came back, which is section 3.7's "no local state": the panel shows the
    // engine's answer, not its own idea of it.
    settings: null,
    // What is TYPED and not yet sent. This is not a second copy of the
    // settings — nothing here is a setting until a PUT has taken it — and it
    // has to live outside the DOM for `engineChoice`'s reason: Status is on a
    // four-second timer that redraws the console, and a half-typed key inside
    // an element would be thrown away under the operator's hands.
    upstreamDraft: {},
    routeDraft: {},
    allowanceDraft: null,
    // What each upstream answered `test` with, so the route pickers can offer
    // real model ids. Not cached across a reload and never written to disk:
    // the list is somebody else's and changes without telling us.
    upstreamModels: {},
    lastStatusAt: 0,
    timer: null
  };

  /** One task's live view, rebuilt from its own event stream. */
  function liveTask(task) {
    return {
      id: task.task_id,
      task: task,
      step: null,
      bytesDone: null,
      bytesTotal: null,
      file: null,
      lines: [],
      skipped: [],
      ended: null
    };
  }

  // ------------------------------------------------------------- refusals

  function Refusal(status, code, message, details) {
    this.status = status;
    this.code = code;
    this.message = message;
    this.details = details;
  }

  Refusal.prototype.toString = function () {
    return this.code + ': ' + this.message;
  };

  async function refusalOf(response) {
    var body = null;
    try {
      body = await response.json();
    } catch (parseFailed) {
      body = null;
    }
    if (body && body.error && body.error.code) {
      return new Refusal(
        response.status,
        body.error.code,
        body.error.message,
        body.error.details === undefined ? null : body.error.details
      );
    }
    // Not a Crucible refusal at all — a proxy, a gateway, something between.
    // Named as what it is rather than dressed up as one of ours.
    return new Refusal(
      response.status,
      'http_' + response.status,
      'this answer did not come from Crucible: HTTP ' +
        response.status +
        ' with no named refusal in it',
      null
    );
  }

  // ------------------------------------------------------------- transport

  function headers(extra) {
    var built = { Authorization: 'Bearer ' + state.token };
    built[API_HEADER] = API_VERSION;
    if (extra) {
      for (var key in extra) {
        if (Object.prototype.hasOwnProperty.call(extra, key)) {
          built[key] = extra[key];
        }
      }
    }
    return built;
  }

  async function call(path, options) {
    var init = options ? Object.assign({}, options) : {};
    init.headers = headers(init.headers);
    var response;
    try {
      response = await fetch(path, init);
    } catch (unreachable) {
      throw new Refusal(
        0,
        'unreachable',
        'the browser could not reach ' +
          path +
          ' on this server: ' +
          unreachable.message,
        null
      );
    }
    if (!response.ok) {
      var refusal = await refusalOf(response);
      if (response.status === 401) {
        signOut(refusal);
      }
      throw refusal;
    }
    if (response.status === 204) {
      return null;
    }
    return response.json();
  }

  // Written as whole template literals rather than concatenated halves so that
  // `tests/test_ui_mount.py` can read every path this page calls straight out
  // of the file and check each one against the app's route table. A path built
  // out of fragments is a path a drift guard cannot see.
  function taskPath(id) {
    var safe = encodeURIComponent(id);
    return `/v1/tasks/${safe}`;
  }

  function taskEventsPath(id) {
    var safe = encodeURIComponent(id);
    return `/v1/tasks/${safe}/events`;
  }

  // ------------------------------------------------------------------ token

  function readStoredToken() {
    try {
      return window.localStorage.getItem(TOKEN_KEY);
    } catch (blocked) {
      return null;
    }
  }

  function storeToken(token) {
    try {
      window.localStorage.setItem(TOKEN_KEY, token);
    } catch (blocked) {
      // A browser that refuses storage still runs the page for this visit; the
      // next reload will ask again, which is the truth rather than a surprise.
    }
  }

  function forgetToken() {
    try {
      window.localStorage.removeItem(TOKEN_KEY);
    } catch (blocked) {
      return;
    }
  }

  /** `#token=<t>` from a pairing line: stored, then taken out of the address. */
  function tokenFromFragment() {
    var hash = window.location.hash;
    if (!hash || hash.length < 2) {
      return null;
    }
    var found = new URLSearchParams(hash.slice(1)).get('token');
    if (!found) {
      return null;
    }
    // Out of the address bar, out of the back button, out of a bookmark. A
    // fragment never reached this server, which is why it travelled in one.
    window.history.replaceState(
      null,
      '',
      window.location.pathname + window.location.search
    );
    return found;
  }

  function signOut(refusal) {
    forgetToken();
    state.token = null;
    stopStream();
    if (state.timer !== null) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    showGate(refusal);
  }

  // ----------------------------------------------------------------- pieces

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      for (var key in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, key)) {
          continue;
        }
        var value = attrs[key];
        if (value === null || value === undefined || value === false) {
          continue;
        }
        if (key === 'class') {
          node.className = value;
        } else if (key === 'text') {
          node.textContent = value;
        } else if (key === 'hidden') {
          node.hidden = true;
        } else if (key === 'disabled') {
          node.disabled = true;
        } else if (key.indexOf('on') === 0) {
          node.addEventListener(key.slice(2), value);
        } else {
          node.setAttribute(key, value);
        }
      }
    }
    if (children) {
      for (var index = 0; index < children.length; index += 1) {
        var child = children[index];
        if (child === null || child === undefined || child === false) {
          continue;
        }
        node.appendChild(
          typeof child === 'string' ? document.createTextNode(child) : child
        );
      }
    }
    return node;
  }

  function chip(text, tone) {
    return el('span', { class: tone ? 'chip ' + tone : 'chip', text: text });
  }

  function fact(term, value) {
    return [el('dt', { text: term }), el('dd', null, [value])];
  }

  function facts(pairs) {
    var list = el('dl', { class: 'facts' });
    for (var index = 0; index < pairs.length; index += 1) {
      var pair = pairs[index];
      if (pair === null) {
        continue;
      }
      var built = fact(pair[0], pair[1]);
      list.appendChild(built[0]);
      list.appendChild(built[1]);
    }
    return list;
  }

  function mono(text) {
    return el('span', { class: 'mono', text: text });
  }

  /**
   * The capabilities this server is serving right now, by name — `llm`,
   * `tts`, … — which is what a job type row and a catalog row both speak.
   * Null when `/v1/info` has not been read or was refused: an empty set would
   * read as "this server serves nothing", which is a different claim.
   */
  function offeredCapabilities() {
    if (state.info === null) {
      return null;
    }
    var names = [];
    for (var index = 0; index < state.info.capabilities.length; index += 1) {
      names.push(state.info.capabilities[index].job_type);
    }
    return names;
  }

  function refusalBox(refusal) {
    if (!refusal) {
      return null;
    }
    var box = el('div', { class: 'refusal', role: 'status' }, [
      el('code', { class: 'refusal-code', text: refusal.code }),
      el('span', { class: 'refusal-message', text: refusal.message })
    ]);
    var details = refusal.details;
    if (details && details.fact && details.who) {
      // The four-shaped `409 server_busy`. `details.who` is the server's own
      // sentence about the holder and it is shown as written: an operator told
      // only "busy" concludes the button is broken and presses it until it is.
      box.appendChild(
        el('span', { class: 'refusal-who' }, [
          el('b', { text: details.fact + ' holds the card: ' }),
          details.who
        ])
      );
    }
    if (details && details.problems && details.problems.length) {
      var problems = el('ul', { class: 'steps' });
      for (var index = 0; index < details.problems.length; index += 1) {
        problems.appendChild(el('li', { text: String(details.problems[index]) }));
      }
      box.appendChild(problems);
    }
    return box;
  }

  function setRefusal(where, refusal) {
    state.refusals[where] = refusal;
  }

  // ---------------------------------------------------------------- numbers

  var UNITS = ['B', 'kB', 'MB', 'GB', 'TB'];

  /**
   * Decimal, because that is what a model card and a download both quote, and
   * `null` is returned as `null` — never 0. A subject whose manifest declares
   * no size is a subject nobody knows the size of, and "0" is a number.
   */
  function bytesText(value) {
    if (value === null || value === undefined) {
      return null;
    }
    var scaled = value;
    var unit = 0;
    while (scaled >= 1000 && unit < UNITS.length - 1) {
      scaled = scaled / 1000;
      unit += 1;
    }
    var places = unit === 0 ? 0 : scaled < 100 ? 1 : 0;
    return scaled.toFixed(places) + ' ' + UNITS[unit];
  }

  function secondsText(value) {
    var whole = Math.floor(value);
    var hours = Math.floor(whole / 3600);
    var minutes = Math.floor((whole % 3600) / 60);
    var seconds = whole % 60;
    if (hours > 0) {
      return hours + ' h ' + minutes + ' min';
    }
    if (minutes > 0) {
      return minutes + ' min ' + seconds + ' s';
    }
    return seconds + ' s';
  }

  function clockText(stamp) {
    if (!stamp) {
      return null;
    }
    var when = new Date(stamp);
    if (isNaN(when.getTime())) {
      return stamp;
    }
    return when.toLocaleTimeString();
  }

  // ------------------------------------------------------------ the reads

  async function loadSetup() {
    try {
      state.setup = await call('/v1/setup');
      setRefusal('setup', null);
    } catch (refusal) {
      state.setup = null;
      setRefusal('setup', refusal);
    }
  }

  async function loadPairingRequests() {
    try {
      state.pairingRequests = (await call('/v1/pairing/requests')).requests;
      setRefusal('pairing', null);
    } catch (refusal) {
      state.pairingRequests = [];
      setRefusal('pairing', refusal);
    }
  }

  async function decidePairing(request, allow) {
    try {
      await call('/v1/pairing/decision', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: request.id, user_code: request.user_code, allow: allow }) });
      await loadPairingRequests();
    } catch (refusal) {
      setRefusal('pairing', refusal);
    }
    renderConnect();
  }

  async function loadInfo() {
    try {
      state.info = await call('/v1/info');
      setRefusal('info', null);
    } catch (refusal) {
      state.info = null;
      setRefusal('info', refusal);
    }
  }

  async function loadActivity() {
    try {
      // No `?accelerator_probe=true`. The probe spawns `nvidia-smi` per read,
      // and this one is on a timer; the route's own ruling is that it is
      // opt-in. The card's identity comes from `/v1/info`, which costs nothing.
      state.activity = await call('/v1/activity');
      setRefusal('activity', null);
    } catch (refusal) {
      state.activity = null;
      setRefusal('activity', refusal);
    }
  }

  async function loadCapability() {
    try {
      state.capability = await call('/v1/capability');
      setRefusal('capability', null);
    } catch (refusal) {
      state.capability = null;
      setRefusal('capability', refusal);
    }
  }

  async function loadSettings() {
    try {
      state.settings = await call('/v1/settings');
      setRefusal('settings', null);
    } catch (refusal) {
      state.settings = null;
      setRefusal('settings', refusal);
    }
  }

  async function loadCatalog() {
    try {
      state.catalog = await call('/v1/catalog');
      setRefusal('catalog', null);
    } catch (refusal) {
      state.catalog = null;
      setRefusal('catalog', refusal);
    }
  }

  async function loadTasks() {
    try {
      var body = await call('/v1/tasks');
      state.tasks = body.tasks;
      setRefusal('tasks', null);
    } catch (refusal) {
      state.tasks = [];
      setRefusal('tasks', refusal);
    }
    var running = null;
    for (var index = 0; index < state.tasks.length; index += 1) {
      if (state.tasks[index].state === 'running') {
        running = state.tasks[index];
      }
    }
    state.running = running;
    if (running === null) {
      stopStream();
      state.live = null;
    } else if (state.live === null || state.live.id !== running.task_id) {
      state.live = liveTask(running);
      watch(running.task_id);
    } else {
      state.live.task = running;
    }
  }

  // --------------------------------------------------------- the event feed

  function stopStream() {
    if (state.stream !== null) {
      state.stream.abort();
      state.stream = null;
    }
  }

  async function watch(id) {
    stopStream();
    var controller = new AbortController();
    state.stream = controller;
    var response;
    try {
      response = await fetch(taskEventsPath(id), {
        headers: headers({ Accept: 'text/event-stream' }),
        signal: controller.signal
      });
    } catch (dropped) {
      return;
    }
    if (!response.ok) {
      var refusal = await refusalOf(response);
      if (response.status === 401) {
        signOut(refusal);
      }
      setRefusal('tasks', refusal);
      render();
      return;
    }
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';
    while (true) {
      var chunk;
      try {
        chunk = await reader.read();
      } catch (dropped) {
        return;
      }
      if (chunk.done) {
        break;
      }
      buffer += decoder.decode(chunk.value, { stream: true });
      var cut = buffer.indexOf('\n\n');
      while (cut !== -1) {
        onFrame(buffer.slice(0, cut), id);
        buffer = buffer.slice(cut + 2);
        cut = buffer.indexOf('\n\n');
      }
    }
    // The stream ends at the task's terminal event; re-read everything so the
    // sections that an install or a pull changed are the ones on screen.
    if (state.token !== null) {
      await refreshAll();
    }
  }

  function onFrame(frame, id) {
    if (state.live === null || state.live.id !== id) {
      return;
    }
    var name = null;
    var payload = null;
    var lines = frame.split('\n');
    for (var index = 0; index < lines.length; index += 1) {
      var line = lines[index];
      if (line.indexOf(':') === 0) {
        continue;
      }
      var mark = line.indexOf(':');
      if (mark === -1) {
        continue;
      }
      var field = line.slice(0, mark);
      var value = line.slice(mark + 1);
      if (value.indexOf(' ') === 0) {
        value = value.slice(1);
      }
      if (field === 'event') {
        name = value;
      } else if (field === 'data') {
        try {
          payload = JSON.parse(value);
        } catch (notJson) {
          payload = null;
        }
      }
    }
    if (name === null) {
      return;
    }
    applyEvent(name, payload === null ? {} : payload);
    render();
    if (TERMINAL.indexOf(name) === -1) {
      touchStatus();
    }
  }

  function applyEvent(name, data) {
    var live = state.live;
    if (name === 'step') {
      live.step = data;
      // A new step is a new denominator: bytes from the last file would read
      // as progress on this one.
      live.bytesDone = null;
      live.bytesTotal = null;
      live.file = null;
    } else if (name === 'progress') {
      if (data.line !== undefined) {
        live.lines.push(data.line);
        if (live.lines.length > INSTALL_LINES_KEPT) {
          live.lines = live.lines.slice(live.lines.length - INSTALL_LINES_KEPT);
        }
      } else {
        live.bytesDone = data.bytes_done;
        live.bytesTotal = data.bytes_total;
        live.file = data.file;
      }
    } else if (name === 'skipped') {
      live.skipped.push(data.reason);
    } else if (TERMINAL.indexOf(name) !== -1) {
      live.ended = { event: name, data: data };
    }
  }

  /** Status is refreshed on an interval AND on every task event, throttled. */
  function touchStatus() {
    var now = Date.now();
    if (now - state.lastStatusAt < STATUS_THROTTLE_MS) {
      return;
    }
    state.lastStatusAt = now;
    loadActivity().then(render);
  }

  // ------------------------------------------------------------ 1. status

  function holderRows(activity) {
    var rows = [];
    if (activity.running && activity.running.length) {
      for (var index = 0; index < activity.running.length; index += 1) {
        var job = activity.running[index];
        var text = job.type + ' — ' + (job.model === null ? 'no model' : job.model);
        text += ', ' + job.status;
        if (job.progress !== null && job.progress !== undefined) {
          text += ', ' + job.progress + '%';
        }
        if (job.client) {
          text += ', for ' + job.client;
        }
        rows.push(['a job', text]);
      }
    }
    if (activity.lease) {
      var lease = activity.lease;
      var line = 'held by ' + lease.client + ' — ' + lease.act;
      line += ', the resident ' + lease.kind;
      line += ', until ' + clockText(lease.expires_at);
      rows.push(['a lease', line]);
    }
    if (activity.claim) {
      rows.push(['the claim', 'held by ' + activity.claim.held_by]);
    }
    if (activity.streaming) {
      var session = activity.streaming;
      var spoken = session.voice + ' (' + session.narrator_engine + ')';
      if (session.client) {
        spoken += ', for ' + session.client;
      }
      spoken += ', open since ' + clockText(session.since);
      rows.push(['a streaming session', spoken]);
    }
    if (activity.chat && activity.chat.in_flight > 0) {
      for (var row = 0; row < activity.chat.rows.length; row += 1) {
        var entry = activity.chat.rows[row];
        var said = entry.act === null ? 'act not stated' : entry.act;
        said += ' — ' + entry.model;
        if (entry.client) {
          said += ', for ' + entry.client;
        }
        rows.push(['a chat', said]);
      }
    }
    return rows;
  }

  /** Is this `cuda-linux` server the WSL guest of a Windows machine?

   * 4.7's Status line says **Engine: WSL2 (vLLM/SGLang)** there, and
   * nothing at all on a real Linux box, where there is no second engine to
   * have moved from. The distinction is not one a Linux server can make
   * about itself — inside the guest, WSL looks like Linux — so it is read
   * off the one thing that differs: the guest's kernel names itself.
   * `/v1/info`'s `host.platform` is `linux` either way, so this asks the
   * BROWSER, which is running on the Windows machine when it is one.
   */
  function isWindowsHost() {
    return (
      typeof navigator !== 'undefined' &&
      typeof navigator.platform === 'string' &&
      navigator.platform.indexOf('Win') === 0
    );
  }

  function renderStatus() {
    var body = document.getElementById('status-body');
    body.textContent = '';

    var setupRefusal = refusalBox(state.refusals.setup);
    if (setupRefusal) {
      body.appendChild(setupRefusal);
    }
    var activityRefusal = refusalBox(state.refusals.activity);
    if (activityRefusal) {
      body.appendChild(activityRefusal);
    }

    var setup = state.setup;
    var activity = state.activity;
    var pairs = [];

    if (setup) {
      pairs.push(['Server', mono(setup.name)]);
      pairs.push([
        'Build',
        el('span', null, [
          'Crucible ',
          el('span', { class: 'num', text: setup.version }),
          ' · backend ',
          mono(setup.backend)
        ])
      ]);
    }

    // WHICH ENGINE RUNS THIS MACHINE, and the one control that changes it
    // (PHASE15-HOST.md 4.7). Shown on a Windows machine only, because on
    // Linux and the Mac there is nothing to move: the backend IS the engine
    // and there is no second one beside it.
    if (setup && setup.backend === 'llama-windows') {
      var engineLine = el('span', null, [
        chip('Windows (llama.cpp)', 'ok'),
        ' — the llm classes and pages. TTS, ASR, alignment, RVC and denoise ' +
          'are Python engines and need the WSL2 guest.'
      ]);
      pairs.push(['Engine', engineLine]);
      pairs.push([
        '',
        el('span', null, [
          el('button', {
            id: 'move-to-wsl',
            class: 'button primary',
            type: 'button',
            disabled: state.running !== null,
            title:
              state.running !== null
                ? 'one operator task at a time; this server is busy'
                : 'install the WSL2 engine, move this config and these ' +
                  'weights into it, and stop the Windows server',
            onclick: function () {
              var question =
                'Move this machine to the WSL2 engine?' +
                '\n\nThis installs the guest, pulls each installed subject ' +
                'in it, deletes the Windows copies once the guest has them, ' +
                'and stops the Windows server. The token does not change, so ' +
                'every app that paired stays paired.' +
                '\n\nIt may need administrator, and it may need a reboot.';
              if (!window.confirm(question)) {
                return;
              }
              // THE SERVER DOES NOT RUN IT. It hands the move to the host's
              // loopback door and relays the host's events under this task
              // id, so what this page watches is the same Tasks panel that
              // watches a pull. A server started by hand refuses
              // `engine_move_needs_host`, and that refusal is drawn like any
              // other.
              submit({ type: 'engine', target: 'wsl' }, 'engine:wsl');
            }
          }, ['Move to WSL2…']),
          ' ',
          el('span', {
            class: 'empty',
            text:
              'faster page reading and text under vLLM/SGLang, and the five ' +
              'Python job types'
          })
        ])
      ]);
    } else if (setup && setup.backend === 'cuda-linux' && isWindowsHost()) {
      pairs.push([
        'Engine',
        el('span', null, [chip('WSL2 (vLLM/SGLang)', 'ok')])
      ]);
    }

    if (state.info && state.info.host && state.info.host.gpu) {
      var gpu = state.info.host.gpu;
      var vram = bytesText(gpu.vram_bytes);
      pairs.push([
        'Card',
        el('span', null, [
          gpu.name,
          ' · ',
          el('span', { class: 'num', text: vram === null ? 'not stated' : vram }),
          ' · ',
          state.info.host.platform + '/' + state.info.host.arch
        ])
      ]);
    }

    if (activity) {
      var resident = activity.resident;
      if (resident === null) {
        pairs.push(['Resident', el('span', { class: 'empty', text: 'nothing is on the card' })]);
      } else {
        var estimate = bytesText(resident.memory_bytes_estimate);
        pairs.push([
          'Resident',
          el('span', null, [
            chip(resident.kind, 'ok'),
            ' ',
            mono(resident.id),
            ' · loaded ' + clockText(resident.since),
            estimate === null ? null : ' · about ' + estimate
          ])
        ]);
      }

      if (activity.warming !== null && activity.warming !== undefined) {
        pairs.push([
          'Warming',
          el('span', null, [chip('loading', 'warn'), ' ', mono(activity.warming)])
        ]);
      }

      var holders = holderRows(activity);
      if (holders.length === 0) {
        pairs.push([
          'Held by',
          el('span', { class: 'empty', text: 'nothing holds the card' })
        ]);
      } else {
        var list = el('ul', { class: 'holders' });
        for (var index = 0; index < holders.length; index += 1) {
          list.appendChild(
            el('li', null, [
              el('span', { class: 'holder-fact', text: holders[index][0] + ' — ' }),
              holders[index][1]
            ])
          );
        }
        pairs.push(['Held by', list]);
      }

      var lane = activity.slots.accelerated;
      pairs.push([
        'Lane',
        el('span', null, [
          el('span', { class: 'num', text: lane.busy + ' of ' + lane.of }),
          ' busy · queue ',
          el('span', { class: 'num', text: String(lane.queue_depth) }),
          ' · ',
          lane.accepts_work
            ? chip('accepts work', 'ok')
            : chip('will refuse a job', 'warn')
        ])
      ]);
      pairs.push([
        'Up',
        el('span', { class: 'num', text: secondsText(activity.server.uptime_s) })
      ]);
    }

    if (setup) {
      pairs.push(['Config', mono(setup.config_path)]);
    }

    if (pairs.length) {
      body.appendChild(facts(pairs));
    } else if (!setupRefusal && !activityRefusal) {
      body.appendChild(el('p', { class: 'empty', text: 'reading…' }));
    }

    renderBar();
  }

  function renderBar() {
    document.getElementById('bar-name').textContent = state.setup
      ? state.setup.name
      : 'operator console';
    var bar = document.getElementById('bar-state');
    bar.textContent = '';
    if (state.running) {
      bar.appendChild(chip(state.running.type + ' running', 'accent'));
    }
    if (state.activity) {
      if (state.activity.resident) {
        bar.appendChild(chip('resident ' + state.activity.resident.id, 'ok'));
      }
      if (state.activity.slots.accelerated.busy > 0) {
        bar.appendChild(chip('lane busy', 'warn'));
      }
    }
    if (bar.children.length === 0 && state.activity) {
      bar.appendChild(chip('idle'));
    }
  }

  // ------------------------------------------------------------- 2. tasks

  /** What the task was ASKED for, from the request the server echoes back. */
  function requestText(task) {
    var request = task.request;
    if (task.type === 'pull') {
      return 'pull ' + request.kind + ' ' + request.id;
    }
    if (task.type === 'install') {
      return (
        'install ' +
        request.job_type +
        (request.narrator_engine ? ' (' + request.narrator_engine + ')' : '')
      );
    }
    if (task.type === 'module') {
      return 'module ' + request.module.name + ' ' + request.module.version;
    }
    return task.type;
  }

  function renderRunning() {
    var live = state.live;
    var task = state.running;
    var box = el('div', { class: 'rows' });
    var head = el('div', { class: 'row head' }, [
      el('span', { text: 'Running' }),
      el('span', { text: 'started ' + clockText(task.started) })
    ]);
    box.appendChild(head);

    var cancel = el('button', {
      // Ids on every control an operator can tab to, so the timed redraw can
      // put the keyboard back where it found it (see `render`).
      id: 'task-cancel',
      class: 'button danger',
      type: 'button',
      onclick: function () {
        cancelTask(task.task_id);
      }
    }, ['Cancel']);

    var row = el('div', { class: 'row' }, [
      el('span', null, [
        el('span', { class: 'row-title', text: requestText(task) }),
        el('span', { class: 'row-id', text: task.task_id })
      ]),
      el('span', null, [live && live.step ? live.step.name : 'starting…']),
      el(
        'span',
        { class: 'row-size' },
        [live && live.step ? live.step.index + ' of ' + live.step.total : '']
      ),
      el('span', { class: 'row-action' }, [cancel])
    ]);
    box.appendChild(row);

    var detail = el('div', { class: 'row wide' });
    var anything = false;

    if (live && live.bytesDone !== null) {
      // "Pulling… 3.2 of 19.4 GB" — the operator's sentence, not the wire's.
      var done = bytesText(live.bytesDone);
      var total = bytesText(live.bytesTotal);
      var text = 'Pulling ' + (live.file ? live.file : 'weights') + ' — ' + done;
      text += total === null ? ' so far, total not declared' : ' of ' + total;
      var bar = el('div', {
        class: live.bytesTotal ? 'bar' : 'bar indeterminate',
        role: 'progressbar',
        'aria-valuetext': text
      }, [el('span')]);
      if (live.bytesTotal) {
        var fraction = Math.min(1, live.bytesDone / live.bytesTotal);
        bar.firstChild.style.width = (fraction * 100).toFixed(1) + '%';
      }
      detail.appendChild(
        el('div', { class: 'progress' }, [
          el('div', { class: 'progress-line', text: text }),
          bar
        ])
      );
      anything = true;
    }

    if (live && live.lines.length) {
      // pip's own lines. R4: they are shown because an operator wants to read
      // them, and nothing here parses one.
      var pane = el('div', {
        class: 'pane',
        role: 'log',
        'aria-label': 'installer output',
        text: live.lines.join('\n')
      });
      detail.appendChild(pane);
      anything = true;
      window.setTimeout(function () {
        pane.scrollTop = pane.scrollHeight;
      }, 0);
    }

    if (live && live.skipped.length) {
      var skipped = el('ul', { class: 'steps' });
      for (var index = 0; index < live.skipped.length; index += 1) {
        skipped.appendChild(
          el('li', null, [
            el('span', { class: 'step-index', text: 'skipped' }),
            live.skipped[index]
          ])
        );
      }
      detail.appendChild(skipped);
      anything = true;
    }

    if (anything) {
      box.appendChild(detail);
    }
    return box;
  }

  function renderTasks() {
    var body = document.getElementById('tasks-body');
    body.textContent = '';

    var refusal = refusalBox(state.refusals.tasks);
    if (refusal) {
      body.appendChild(refusal);
    }
    var cancelled = refusalBox(state.refusals.cancel);
    if (cancelled) {
      body.appendChild(cancelled);
    }

    if (state.running) {
      body.appendChild(renderRunning());
    } else {
      body.appendChild(
        el('p', { class: 'empty', text: 'No task is running on this server.' })
      );
    }

    var finished = [];
    for (var index = 0; index < state.tasks.length; index += 1) {
      if (state.tasks[index].state !== 'running') {
        finished.push(state.tasks[index]);
      }
    }
    if (finished.length === 0) {
      return;
    }

    body.appendChild(
      el('p', { class: 'subhead', text: 'Finished' }, null)
    );
    var rows = el('div', { class: 'rows' });
    var shown = finished.slice(0, 8);
    for (var row = 0; row < shown.length; row += 1) {
      var task = shown[row];
      var tone = task.state === 'done' ? 'ok' : task.state === 'failed' ? 'bad' : '';
      var cells = el('div', { class: 'row' }, [
        el('span', null, [
          el('span', { class: 'row-title', text: requestText(task) }),
          el('span', { class: 'row-id', text: task.task_id })
        ]),
        el('span', { class: 'row-size', text: clockText(task.finished) }),
        el('span'),
        el('span', { class: 'row-action' }, [chip(task.state, tone)])
      ]);
      if (task.error) {
        // The failure verbatim: `pull_failed` is the weights module's own
        // sentence, `install_failed` the console script's exit.
        cells.appendChild(
          el('div', { class: 'row-span' }, [
            refusalBox(new Refusal(0, task.error.code, task.error.message, null))
          ])
        );
      }
      rows.appendChild(cells);
    }
    body.appendChild(rows);
    if (finished.length > shown.length) {
      body.appendChild(
        el('p', {
          class: 'note',
          text:
            'and ' +
            (finished.length - shown.length) +
            ' older, which this server keeps in memory only'
        })
      );
    }
  }

  // --------------------------------------------------------- 3. job types

  function renderJobTypes() {
    var body = document.getElementById('types-body');
    body.textContent = '';

    var refusal = refusalBox(state.refusals.capability);
    if (refusal) {
      body.appendChild(refusal);
      body.appendChild(
        el('p', {
          class: 'note',
          text:
            'Nothing is listed rather than some of it: a server that has ' +
            'decided nothing about its card cannot say what it could hold.'
        })
      );
      return;
    }
    var infoRefusal = refusalBox(state.refusals.info);
    if (infoRefusal) {
      body.appendChild(infoRefusal);
    }
    var offered = offeredCapabilities();
    if (state.capability === null || offered === null) {
      body.appendChild(el('p', { class: 'empty', text: 'reading…' }));
      return;
    }

    var verdicts = {};
    for (var index = 0; index < state.capability.classes.length; index += 1) {
      var verdict = state.capability.classes[index];
      verdicts[verdict.capability] = verdict;
    }

    var card = bytesText(state.capability.total_bytes);
    var reserve = bytesText(state.capability.desktop_allowance_bytes);
    body.appendChild(
      el('p', { class: 'lead' }, [
        'Decided on a ',
        el('span', { class: 'num', text: card === null ? 'card of unstated size' : card }),
        ' card with ',
        el('span', { class: 'num', text: reserve === null ? 'nothing' : reserve }),
        ' held back for the desktop. A class that is off names the number that ' +
          'turned it off.'
      ])
    );

    var rows = el('div', { class: 'rows' });
    for (var slot = 0; slot < state.capability.job_types.length; slot += 1) {
      rows.appendChild(jobTypeRow(state.capability.job_types[slot], verdicts, offered));
    }
    body.appendChild(rows);
  }

  function jobTypeRow(entry, verdicts, offered) {
    var here = offered.indexOf(entry.job_type) !== -1;
    var block = el('div', { class: 'row' });

    block.appendChild(
      el('span', null, [
        el('span', { class: 'row-title', text: entry.job_type }),
        el('span', {
          class: 'row-id',
          text: entry.classes.join(', ')
        })
      ])
    );

    var verdictList = el('span', { class: 'chips' });
    for (var index = 0; index < entry.classes.length; index += 1) {
      var verdict = verdicts[entry.classes[index]];
      if (verdict === undefined) {
        continue;
      }
      verdictList.appendChild(
        chip(verdict.capability, verdict.enabled ? 'ok' : 'warn')
      );
    }
    block.appendChild(verdictList);

    block.appendChild(
      el('span', { class: 'row-size' }, [
        here ? chip('installed', 'ok') : chip('not installed')
      ])
    );

    block.appendChild(el('span', { class: 'row-action' }, [installControl(entry, here)]));

    var reasons = el('ul', { class: 'steps' });
    for (var row = 0; row < entry.classes.length; row += 1) {
      var each = verdicts[entry.classes[row]];
      if (each === undefined) {
        continue;
      }
      var line = each.capability + ': ' + each.reason;
      if (each.selected) {
        line += ' (selected ' + each.selected + ')';
      }
      reasons.appendChild(
        el('li', null, [
          el('span', {
            class: 'step-index',
            text: each.enabled ? 'on ' : 'off'
          }),
          line
        ])
      );
    }
    block.appendChild(el('div', { class: 'row-span' }, [reasons]));

    var note = state.refusals['install:' + entry.job_type];
    if (note) {
      block.appendChild(el('div', { class: 'row-span' }, [refusalBox(note)]));
    }
    return block;
  }

  function installControl(entry, here) {
    if (here) {
      return el('span', {
        class: 'note',
        text: 'this server offers it'
      });
    }
    if (entry.installer === null) {
      return el('span', { class: 'note', text: 'compiled in; nothing installs it' });
    }
    if (entry.installer !== entry.job_type) {
      return el('span', {
        class: 'note',
        text: 'built by installing ' + entry.installer
      });
    }

    var wants = entry.narrator_engines.length > 0;
    var group = el('span', { class: 'controls' });
    if (wants) {
      // Which engine, from the server's list. The choice is kept in `state`
      // because the console is redrawn on a timer and a value that lived only
      // in the element would reset under the operator's hand.
      if (state.engineChoice[entry.job_type] === undefined) {
        state.engineChoice[entry.job_type] = entry.narrator_engines[0];
      }
      var select = el('select', {
        id: 'engine-' + entry.job_type,
        'aria-label': 'narrator engine for ' + entry.job_type,
        onchange: function (event) {
          state.engineChoice[entry.job_type] = event.target.value;
        }
      });
      for (var index = 0; index < entry.narrator_engines.length; index += 1) {
        select.appendChild(
          el('option', {
            value: entry.narrator_engines[index],
            text: entry.narrator_engines[index]
          })
        );
      }
      select.value = state.engineChoice[entry.job_type];
      group.appendChild(select);
    }

    group.appendChild(
      el('button', {
        id: 'install-' + entry.job_type,
        class: 'button primary',
        type: 'button',
        disabled: state.running !== null,
        onclick: function () {
          var request = { type: 'install', job_type: entry.job_type };
          if (wants) {
            request.narrator_engine = state.engineChoice[entry.job_type];
          }
          submit(request, 'install:' + entry.job_type);
        }
      }, ['Install'])
    );
    return group;
  }

  // ---------------------------------------------------------- 4. settings
  //
  // PHASE15-HOST.md section 3.7. Owen, 2026-09-14: *"Settings live in the
  // engine and nowhere else."* This panel and BookForge's settings section
  // and Foundry's card are three windows onto ONE store, which is why every
  // control here is a `PUT /v1/settings` and why the panel redraws from the
  // document the PUT hands back rather than from anything it remembers.

  var UPSTREAM_LABELS = {
    anthropic: 'Anthropic',
    openai: 'OpenAI',
    ollama: 'Ollama'
  };

  var UPSTREAM_ORDER = ['anthropic', 'openai', 'ollama'];

  /** The field each upstream takes. The server owns this; the page reads it. */
  function upstreamField(name) {
    return name === 'ollama' ? 'url' : 'key';
  }

  /**
   * Send one patch and take what comes back as the truth.
   *
   * A refusal is shown WHERE IT WAS EARNED — `details.field` is the dotted
   * path the server refused, so the control that caused it is the one that
   * says so, rather than a banner at the top of a panel with nine controls.
   */
  async function putSettings(patch, where) {
    try {
      state.settings = await call('/v1/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch)
      });
      setRefusal(where, null);
      return true;
    } catch (refusal) {
      setRefusal(where, refusal);
      return false;
    } finally {
      // Capability and the catalog both move when a route does, and the
      // panels above this one draw from them. Re-read rather than patched in
      // place: the server recomputed the record, and guessing what it decided
      // would be a second opinion about the same write.
      await loadCapability();
      render();
    }
  }

  function renderSettings() {
    var body = document.getElementById('settings-body');
    body.textContent = '';

    var refusal = refusalBox(state.refusals.settings);
    if (refusal) {
      body.appendChild(refusal);
    }
    if (state.settings === null) {
      if (!refusal) {
        body.appendChild(el('p', { class: 'empty', text: 'reading…' }));
      }
      return;
    }

    body.appendChild(
      el('p', { class: 'lead' }, [
        'Where each kind of text work runs, and the accounts this engine may ' +
          'spend. BookForge and Foundry draw these same rows — there is one ' +
          'store and this is it.'
      ])
    );

    body.appendChild(renderRouteRows());
    body.appendChild(renderUpstreamCards());
    body.appendChild(renderAllowance());
  }

  /** Every distinct upstream model something is already routed to. */
  function routedModels() {
    var found = [];
    var routes = state.settings.routes;
    for (var name in routes) {
      if (!Object.prototype.hasOwnProperty.call(routes, name)) {
        continue;
      }
      var row = routes[name];
      if (row.route === 'upstream' && found.indexOf(row.model) < 0) {
        found.push(row.model);
      }
    }
    return found;
  }

  /** The ids `test` reported, as `<upstream>/<model>`, for configured ones. */
  function testedModels() {
    var found = [];
    for (var index = 0; index < UPSTREAM_ORDER.length; index += 1) {
      var name = UPSTREAM_ORDER[index];
      if (!state.settings.upstreams[name].configured) {
        continue;
      }
      var listed = state.upstreamModels[name];
      if (!listed) {
        continue;
      }
      for (var i = 0; i < listed.length; i += 1) {
        found.push(name + '/' + listed[i]);
      }
    }
    return found;
  }

  function renderRouteRows() {
    var block = el('div', { class: 'block' }, [
      el('p', { class: 'subhead', text: 'Where the work runs' })
    ]);
    var routes = state.settings.routes;
    var names = Object.keys(routes).sort();
    for (var index = 0; index < names.length; index += 1) {
      block.appendChild(routeRow(names[index], routes[names[index]]));
    }
    return block;
  }

  function routeRow(name, row) {
    var options = [];
    // `local` always, and it says WHAT local means on this machine, so the
    // choice is between two named things rather than between a word and a
    // word (3.7). `nothing fits` is an answer, not an absence.
    options.push({
      value: 'local',
      label: 'local — ' + (row.model === null ? 'nothing fits' : row.model)
    });
    var offered = routedModels();
    var tested = testedModels();
    for (var i = 0; i < tested.length; i += 1) {
      if (offered.indexOf(tested[i]) < 0) {
        offered.push(tested[i]);
      }
    }
    if (row.route === 'upstream' && offered.indexOf(row.model) < 0) {
      offered.push(row.model);
    }
    for (var j = 0; j < offered.length; j += 1) {
      options.push({ value: offered[j], label: offered[j] });
    }
    options.push({ value: '', label: 'an upstream model…' });

    var current = row.route === 'upstream' ? row.model : 'local';
    var typing = Object.prototype.hasOwnProperty.call(state.routeDraft, name);
    var select = el('select', {
      id: 'route-' + name,
      'aria-label': 'where ' + name + ' runs',
      onchange: function (event) {
        var picked = event.target.value;
        if (picked === '') {
          state.routeDraft[name] = '';
          renderSettings();
          return;
        }
        delete state.routeDraft[name];
        var patch = { routes: {} };
        patch.routes[name] = picked;
        putSettings(patch, 'settings');
      }
    });
    for (var k = 0; k < options.length; k += 1) {
      var option = el('option', {
        value: options[k].value,
        text: options[k].label
      });
      if (!typing && options[k].value === current) {
        option.selected = true;
      }
      if (typing && options[k].value === '') {
        option.selected = true;
      }
      select.appendChild(option);
    }

    var pieces = [select];
    if (typing) {
      var free = el('input', {
        id: 'route-free-' + name,
        type: 'text',
        spellcheck: 'false',
        placeholder: 'anthropic/claude-sonnet-5',
        'aria-label': 'an upstream model id for ' + name,
        oninput: function (event) {
          state.routeDraft[name] = event.target.value;
        }
      });
      free.value = state.routeDraft[name];
      pieces.push(free);
      pieces.push(
        el('button', {
          class: 'button quiet',
          type: 'button',
          onclick: function () {
            var typed = (state.routeDraft[name] || '').trim();
            if (typed === '') {
              return;
            }
            var patch = { routes: {} };
            patch.routes[name] = typed;
            putSettings(patch, 'settings').then(function (took) {
              if (took) {
                delete state.routeDraft[name];
                renderSettings();
              }
            });
          }
        }, ['Route'])
      );
    }

    return el('div', { class: 'setting' }, [
      el('div', { class: 'setting-head' }, [
        el('span', { class: 'setting-name', text: name }),
        chip(row.route, row.route === 'upstream' ? 'warn' : 'ok')
      ]),
      el('div', { class: 'setting-actions' }, pieces)
    ]);
  }

  function renderUpstreamCards() {
    var block = el('div', { class: 'block' }, [
      el('p', { class: 'subhead', text: 'Accounts this engine may spend' }),
      el('p', { class: 'note' }, [
        'A key is write-only: it is stored at mode 0600 beside this server’s ' +
          'token and never read back. What is shown is its last four ' +
          'characters, which is enough to recognise which one is there.'
      ])
    ]);
    for (var index = 0; index < UPSTREAM_ORDER.length; index += 1) {
      block.appendChild(upstreamCard(UPSTREAM_ORDER[index]));
    }
    return block;
  }

  function upstreamCard(name) {
    var entry = state.settings.upstreams[name];
    var field = upstreamField(name);
    var draft = state.upstreamDraft[name] || '';

    var input = el('input', {
      id: 'upstream-' + name,
      type: field === 'key' ? 'password' : 'text',
      spellcheck: 'false',
      autocomplete: 'off',
      placeholder: field === 'key' ? 'paste a key' : 'http://host:11434',
      'aria-label': UPSTREAM_LABELS[name] + ' ' + field,
      oninput: function (event) {
        state.upstreamDraft[name] = event.target.value;
      }
    });
    input.value = draft;

    var probe = function () {
      var typed = draft.trim();
      var body = typed === '' ? {} : {};
      if (typed !== '') {
        body[field] = typed;
      }
      return body;
    };

    var testButton = el('button', {
      class: 'button quiet',
      type: 'button',
      onclick: async function () {
        try {
          var answer = await call(
            settingsTestPath(name),
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(probe())
            }
          );
          state.upstreamModels[name] = answer.models;
          setRefusal('upstream-' + name, null);
        } catch (refusal) {
          // The three test refusals are ANSWERS to "does this work", so they
          // are shown on this card rather than thrown at the panel. They are
          // shown verbatim, because the words are the provider's.
          delete state.upstreamModels[name];
          setRefusal('upstream-' + name, refusal);
        }
        renderSettings();
      }
    }, ['Test']);

    var saveButton = el('button', {
      class: 'button primary',
      type: 'button',
      onclick: function () {
        var typed = draft.trim();
        if (typed === '') {
          return;
        }
        var patch = { upstreams: {} };
        patch.upstreams[name] = {};
        patch.upstreams[name][field] = typed;
        putSettings(patch, 'upstream-' + name).then(function (took) {
          if (took) {
            // The field is emptied on success and only on success: a key that
            // was refused is still the one the operator has in their hand.
            delete state.upstreamDraft[name];
            renderSettings();
          }
        });
      }
    }, ['Save']);

    var actions = [input, testButton, saveButton];
    if (entry.configured) {
      actions.push(
        el('button', {
          class: 'button quiet',
          type: 'button',
          onclick: function () {
            var patch = { upstreams: {} };
            patch.upstreams[name] = null;
            putSettings(patch, 'upstream-' + name);
          }
        }, ['Remove'])
      );
    }

    var head = [
      el('span', { class: 'setting-name', text: UPSTREAM_LABELS[name] })
    ];
    if (entry.configured) {
      head.push(chip('configured', 'ok'));
      head.push(mono(field === 'key' ? entry.key_hint : entry.url));
    } else {
      head.push(chip('not configured'));
    }

    var card = el('div', { class: 'setting' }, [
      el('div', { class: 'setting-head' }, head),
      el('div', { class: 'setting-actions' }, actions)
    ]);

    var listed = state.upstreamModels[name];
    if (listed) {
      card.appendChild(
        el('p', { class: 'note' }, [
          listed.length === 0
            ? UPSTREAM_LABELS[name] + ' answered, and lists no models.'
            : UPSTREAM_LABELS[name] +
              ' lists: ' +
              listed.join(', ') +
              '. Pick one in a row above.'
        ])
      );
    }
    var earned = refusalBox(state.refusals['upstream-' + name]);
    if (earned) {
      card.appendChild(earned);
    }
    return card;
  }

  // Written whole so `tests/test_ui_mount.py` can read it: a path built out
  // of fragments is a path the drift guard cannot see.
  function settingsTestPath(name) {
    var safe = encodeURIComponent(name);
    return `/v1/settings/upstreams/${safe}/test`;
  }

  function renderAllowance() {
    var current = state.settings.desktop_allowance_bytes;
    var draft =
      state.allowanceDraft === null ? String(current) : state.allowanceDraft;
    var input = el('input', {
      id: 'allowance',
      type: 'text',
      inputmode: 'numeric',
      spellcheck: 'false',
      'aria-label': 'desktop allowance in bytes',
      oninput: function (event) {
        state.allowanceDraft = event.target.value;
      }
    });
    input.value = draft;
    return el('div', { class: 'block' }, [
      el('p', { class: 'subhead', text: 'Desktop allowance' }),
      el('p', { class: 'note' }, [
        'VRAM this host’s own desktop holds that is not anybody’s job. It is ' +
          'subtracted before a capability is decided, so changing it decides ' +
          'them again — currently ' +
          bytesText(current) +
          '.'
      ]),
      el('div', { class: 'setting-actions' }, [
        input,
        el('button', {
          class: 'button quiet',
          type: 'button',
          onclick: function () {
            var typed = (state.allowanceDraft === null
              ? String(current)
              : state.allowanceDraft
            ).trim();
            if (!/^[0-9]+$/.test(typed)) {
              setRefusal(
                'settings',
                new Refusal(
                  0,
                  'invalid_bytes',
                  'the desktop allowance is a whole number of bytes, and ' +
                    JSON.stringify(typed) +
                    ' is not one. Nothing was sent.',
                  null
                )
              );
              renderSettings();
              return;
            }
            putSettings(
              { desktop_allowance_bytes: Number(typed) },
              'settings'
            ).then(function (took) {
              if (took) {
                state.allowanceDraft = null;
                renderSettings();
              }
            });
          }
        }, ['Set'])
      ])
    ]);
  }

  // ----------------------------------------------------------- 5. catalog

  function renderCatalog() {
    var body = document.getElementById('catalog-body');
    body.textContent = '';
    document.getElementById('catalog-stamp').textContent = '';

    var refusal = refusalBox(state.refusals.catalog);
    if (refusal) {
      body.appendChild(refusal);
      return;
    }
    if (state.catalog === null) {
      body.appendChild(el('p', { class: 'empty', text: 'reading…' }));
      return;
    }

    var rows = state.catalog.rows;
    var installed = 0;
    for (var count = 0; count < rows.length; count += 1) {
      if (rows[count].installed) {
        installed += 1;
      }
    }
    document.getElementById('catalog-stamp').textContent =
      installed + ' of ' + rows.length + ' installed';

    // Grouped by kind in the order the route lists them, which IS the
    // catalog's order — no sort of our own, and no list of kinds here.
    var order = [];
    var grouped = {};
    for (var index = 0; index < rows.length; index += 1) {
      var kind = rows[index].kind;
      if (grouped[kind] === undefined) {
        grouped[kind] = [];
        order.push(kind);
      }
      grouped[kind].push(rows[index]);
    }

    for (var group = 0; group < order.length; group += 1) {
      var name = order[group];
      var box = el('div', { class: 'rows' });
      box.appendChild(
        el('div', { class: 'row head' }, [
          el('span', { text: name }),
          el('span', { class: 'num', text: String(grouped[name].length) })
        ])
      );
      for (var row = 0; row < grouped[name].length; row += 1) {
        box.appendChild(catalogRow(grouped[name][row]));
      }
      var block = el('div', { class: 'block' }, [box]);
      body.appendChild(block);
    }
  }

  async function removeSubject(row) {
    // DELETE, then re-read. The page holds no local state about what is
    // installed (`render` draws `state.catalog` and nothing else), so the
    // re-read is not a refresh for tidiness — it IS how the row learns what
    // happened, and it is what shows a `subject_in_use` refusal beside a row
    // that is still there.
    var path =
      '/v1/catalog/' + encodeURIComponent(row.kind) + '/' + encodeURIComponent(row.id);
    try {
      await call(path, { method: 'DELETE' });
      setRefusal('catalog', null);
    } catch (refusal) {
      setRefusal('catalog', refusal);
    }
    await loadCatalog();
    render();
  }

  function catalogRow(row) {
    var block = el('div', { class: 'row' });

    var title = el('span', null, [
      el('span', { class: 'row-title', text: row.name === null ? row.id : row.name }),
      el('span', { class: 'row-id', text: row.id })
    ]);
    block.appendChild(title);

    var middle = el('span', { class: 'chips' });
    for (var index = 0; index < row.floors.length; index += 1) {
      middle.appendChild(chip('minimum for ' + row.floors[index], 'floor'));
    }
    if (row.license) {
      middle.appendChild(chip(row.license, 'floor'));
    }
    var offered = offeredCapabilities();
    if (offered !== null && offered.indexOf(row.job_type) === -1) {
      // The row's own job type is not served here. Said on the row by
      // comparing against the capabilities `/v1/info` reports, which is that
      // fact's only owner — the catalog deliberately carries no
      // `env_installed` per subject, because that is a fact about the job
      // type and one copy per subject is how it would disagree with itself.
      // You may pull the weights first; the row says what is still missing.
      middle.appendChild(chip(row.job_type + ' not installed', 'warn'));
    }
    block.appendChild(middle);

    var size = row.installed ? bytesText(row.installed_bytes) : bytesText(row.expected_bytes);
    block.appendChild(
      size === null
        ? el('span', { class: 'row-size unknown', text: 'size not declared' })
        : el('span', {
            class: 'row-size',
            text: row.installed ? size : size + ' to fetch'
          })
    );

    var action = el('span', { class: 'row-action' });
    if (row.resident) {
      action.appendChild(chip('resident', 'ok'));
    }
    if (row.installed) {
      action.appendChild(chip('installed', 'ok'));
      // Greyed, not offered: the API refuses a pull of an installed subject
      // `already_installed`, and a button that can only be refused is a
      // button that teaches somebody the page is broken.
      action.appendChild(
        el('button', {
          id: 'pull-' + row.kind + '-' + row.id,
          class: 'button',
          type: 'button',
          disabled: true,
          title:
            'installed. A pull of an installed subject is refused ' +
            'already_installed; re-fetch it deliberately on the server with ' +
            '--force'
        }, ['Pull'])
      );
      // REMOVE, on installed rows only (PHASE15-HOST.md 3.5a). Offered
      // rather than greyed, unlike Pull above, because there IS something
      // this can do — and refused by name when there is not, which is the
      // case the confirmation below cannot know about: a resident model, an
      // open lease, a task naming it.
      //
      // ONE CONFIRMATION, and it names the bytes. This is the only act on
      // this page nobody can undo: everything else either downloads
      // something again or restarts something. `confirm` and not a modal of
      // our own, for the page's whole reason — no build step, no framework,
      // and a browser's own dialog is the one thing every browser draws the
      // same.
      action.appendChild(
        el('button', {
          id: 'remove-' + row.kind + '-' + row.id,
          class: 'button',
          type: 'button',
          disabled: row.resident || state.running !== null,
          title: row.resident
            ? 'it is on the card right now; unload it first'
            : 'delete this subject\'s files from this server',
          onclick: function () {
            var size = bytesText(row.installed_bytes);
            var what = (row.name === null ? row.id : row.name) + ' (' + row.kind + ')';
            var question =
              'Remove ' + what + ' from this server?' +
              (size === null ? '' : ' This frees ' + size + '.') +
              '\n\nThe files are deleted. Getting them back is another download.';
            if (!window.confirm(question)) {
              return;
            }
            removeSubject(row);
          }
        }, ['Remove'])
      );
    } else {
      action.appendChild(
        el('button', {
          id: 'pull-' + row.kind + '-' + row.id,
          class: 'button primary',
          type: 'button',
          disabled: state.running !== null,
          onclick: function () {
            submit(
              { type: 'pull', kind: row.kind, id: row.id },
              'pull:' + row.kind + ':' + row.id
            );
          }
        }, ['Pull'])
      );
    }
    block.appendChild(action);

    var mine =
      state.running !== null &&
      state.running.type === 'pull' &&
      state.running.request.kind === row.kind &&
      state.running.request.id === row.id;
    if (mine && state.live) {
      var done = bytesText(state.live.bytesDone);
      var total = bytesText(state.live.bytesTotal);
      var text =
        done === null
          ? 'Pulling…'
          : 'Pulling… ' + done + (total === null ? ' so far' : ' of ' + total);
      var bar = el('div', {
        class: state.live.bytesTotal ? 'bar' : 'bar indeterminate',
        role: 'progressbar',
        'aria-valuetext': text
      }, [el('span')]);
      if (state.live.bytesTotal) {
        bar.firstChild.style.width =
          ((state.live.bytesDone / state.live.bytesTotal) * 100).toFixed(1) + '%';
      }
      block.appendChild(
        el('div', { class: 'row-span' }, [
          el('div', { class: 'progress' }, [
            el('div', { class: 'progress-line', text: text }),
            bar
          ])
        ])
      );
    }

    var refusal = state.refusals['pull:' + row.kind + ':' + row.id];
    if (refusal) {
      block.appendChild(el('div', { class: 'row-span' }, [refusalBox(refusal)]));
    }

    var source = el('div', { class: 'row-span' }, [
      el('span', { class: 'row-id', text: row.source })
    ]);
    block.appendChild(source);
    return block;
  }

  // ----------------------------------------------------------- 6. connect

  function copyButton(label, text, id) {
    var button = el('button', {
      id: id,
      class: 'button quiet',
      type: 'button'
    }, [label]);
    button.addEventListener('click', function () {
      if (!navigator.clipboard) {
        button.textContent = 'select it and copy';
        return;
      }
      navigator.clipboard.writeText(text).then(
        function () {
          button.textContent = 'copied';
          window.setTimeout(function () {
            button.textContent = label;
          }, 1400);
        },
        function (denied) {
          button.textContent = 'copy refused: ' + denied.message;
        }
      );
    });
    return button;
  }

  function lineItem(text, label, id) {
    return el('div', { class: 'line-item' }, [
      el('span', { class: 'line-text', text: text }),
      el('span', { class: 'line-actions' }, [copyButton(label, text, id)])
    ]);
  }

  function renderConnect() {
    var body = document.getElementById('connect-body');
    body.textContent = '';

    var refusal = refusalBox(state.refusals.setup);
    if (refusal) {
      body.appendChild(refusal);
      return;
    }
    if (state.setup === null) {
      body.appendChild(el('p', { class: 'empty', text: 'reading…' }));
      return;
    }
    var setup = state.setup;

    body.appendChild(el('p', { class: 'lead', text:
      'Apps on this computer connect automatically. On another computer, enter this computer’s IP or hostname in BookForge or Foundry, then approve its matching code here.' }));
    var pairingRefusal = refusalBox(state.refusals.pairing);
    if (pairingRefusal) body.appendChild(pairingRefusal);
    state.pairingRequests.forEach(function (request) {
      var card = el('div', { class: 'line-item' }, [
        el('div', { class: 'line-text' }, [
          el('strong', { text: request.client_name + ' — ' + request.user_code }),
          el('p', { text: 'From ' + request.address + '. Approve only if this code matches the app you are connecting. This grants access to this Crucible.' })
        ]),
        el('div', { class: 'line-actions' }, [
          el('button', { class: 'button', type: 'button', text: 'Approve', onclick: function () { decidePairing(request, true); } }),
          el('button', { class: 'button quiet', type: 'button', text: 'Deny', onclick: function () { decidePairing(request, false); } })
        ])
      ]);
      body.appendChild(card);
    });

    body.appendChild(
      el('p', { class: 'lead' }, [
        'You can also paste a connection line below into BookForge or Foundry → Settings → Crucible ' +
          'Servers → Add. It carries the name, the address and the token, so ' +
          'nobody types a secret twice.'
      ])
    );

    var lines = el('div', { class: 'line-list' });
    for (var index = 0; index < setup.pairing.length; index += 1) {
      lines.appendChild(
        lineItem(setup.pairing[index], 'Copy line', 'copy-pairing-' + index)
      );
    }
    body.appendChild(lines);

    var masked = el('span', {
      class: 'line-text',
      text: state.revealToken ? setup.token : maskOf(setup.token)
    });
    var reveal = el('button', {
      id: 'token-reveal',
      class: 'button quiet',
      type: 'button',
      'aria-pressed': state.revealToken ? 'true' : 'false',
      onclick: function () {
        state.revealToken = !state.revealToken;
        renderConnect();
      }
    }, [state.revealToken ? 'Hide' : 'Reveal']);

    var detail = el('div', { class: 'block' }, [
      el('p', { class: 'subhead', text: 'By hand' }),
      facts([
        ['Name', mono(setup.name)],
        [
          'Addresses',
          el('span', null, [setup.urls.join('  ·  ')])
        ],
        [
          'Token',
          el('span', { class: 'line-item' }, [
            masked,
            el('span', { class: 'line-actions' }, [
              reveal,
              copyButton('Copy token', setup.token, 'copy-token')
            ])
          ])
        ]
      ])
    ]);
    body.appendChild(detail);

    body.appendChild(renderModuleBox());
  }

  function maskOf(token) {
    var dots = '';
    for (var index = 0; index < token.length; index += 1) {
      dots += '•';
    }
    return dots;
  }

  function renderModuleBox() {
    var area = el('textarea', {
      id: 'module-json',
      spellcheck: 'false',
      'aria-label': 'module JSON',
      placeholder:
        'Paste an app’s module JSON here, or drop its .module.json file.',
      oninput: function (event) {
        state.moduleText = event.target.value;
      }
    });
    area.value = state.moduleText;

    var zone = el('div', { class: 'dropzone' }, [area]);
    zone.addEventListener('dragover', function (event) {
      event.preventDefault();
      zone.classList.add('over');
    });
    zone.addEventListener('dragleave', function () {
      zone.classList.remove('over');
    });
    zone.addEventListener('drop', function (event) {
      event.preventDefault();
      zone.classList.remove('over');
      var file = event.dataTransfer.files[0];
      if (!file) {
        return;
      }
      file.text().then(function (text) {
        state.moduleText = text;
        area.value = text;
      });
    });

    var post = el('button', {
      id: 'module-post',
      class: 'button primary',
      type: 'button',
      disabled: state.running !== null,
      onclick: function () {
        var parsed;
        try {
          parsed = JSON.parse(state.moduleText);
        } catch (notJson) {
          // Named before the wire, because this one is the browser's finding
          // and not the server's; the server's own `invalid_module` looks
          // different and says different things.
          setRefusal(
            'module',
            new Refusal(
              0,
              'invalid_json',
              'this is not JSON, so nothing was sent: ' + notJson.message,
              null
            )
          );
          render();
          return;
        }
        submit({ type: 'module', module: parsed }, 'module');
      }
    }, ['Post module']);

    var block = el('div', { class: 'block' }, [
      el('p', { class: 'subhead', text: 'Module' }),
      el('p', { class: 'lead' }, [
        'An app states what it needs from a server in a module file it ' +
          'generates. Posting one installs the job types and pulls the ' +
          'subjects it names, skipping whatever is already here.'
      ]),
      zone,
      el('div', { class: 'controls' }, [post])
    ]);
    var refusal = refusalBox(state.refusals.module);
    if (refusal) {
      block.appendChild(refusal);
    }
    return block;
  }

  // ----------------------------------------------------------- 7. service

  var SERVICE_COMMANDS = [
    ['crucible service status', 'is it installed, and is it up'],
    ['crucible service start', 'start it now'],
    ['crucible service stop', 'stop it'],
    ['crucible service install', 'have the machine run it at boot'],
    ['crucible service uninstall', 'stop having it do that'],
    ['crucible token --url', 'print the pairing lines again']
  ];

  function renderService() {
    var body = document.getElementById('service-body');
    body.textContent = '';

    if (state.setup === null) {
      var refusal = refusalBox(state.refusals.setup);
      body.appendChild(refusal ? refusal : el('p', { class: 'empty', text: 'reading…' }));
      return;
    }

    // `/v1/info` carries service facts on a build that has them; this one does
    // not, so what is drawn is the terminal a person would use instead. The
    // check is on the read, never on a version number.
    var service = state.info && state.info.service ? state.info.service : null;
    if (service !== null) {
      var pairs = [];
      for (var key in service) {
        if (Object.prototype.hasOwnProperty.call(service, key)) {
          pairs.push([key, mono(String(service[key]))]);
        }
      }
      body.appendChild(facts(pairs));
    } else {
      body.appendChild(
        el('p', { class: 'lead' }, [
          'This server does not report how it is being run, so here is what ' +
            'runs it. These are typed on the server itself — the page cannot ' +
            'stop a server it is served by.'
        ])
      );
      var commands = el('pre', { class: 'commands' });
      for (var index = 0; index < SERVICE_COMMANDS.length; index += 1) {
        commands.appendChild(
          el('span', null, [
            el('b', { text: SERVICE_COMMANDS[index][0] }),
            '   # ' + SERVICE_COMMANDS[index][1] + '\n'
          ])
        );
      }
      body.appendChild(commands);
    }

    body.appendChild(
      el('div', { class: 'block' }, [
        facts([
          ['Config', mono(state.setup.config_path)],
          ['Bound to', mono(state.setup.bind)]
        ])
      ])
    );

    var lines = el('div', { class: 'line-list' });
    for (var line = 0; line < state.setup.pairing.length; line += 1) {
      lines.appendChild(
        lineItem(state.setup.pairing[line], 'Copy line', 'copy-service-' + line)
      );
    }
    body.appendChild(
      el('div', { class: 'block' }, [
        el('p', { class: 'subhead', text: 'Pairing lines' }),
        lines
      ])
    );
  }

  // ------------------------------------------------------------- the doors

  async function submit(request, where) {
    setRefusal(where, null);
    render();
    try {
      await call('/v1/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
    } catch (refusal) {
      setRefusal(where, refusal);
      render();
      return;
    }
    await loadTasks();
    render();
  }

  async function cancelTask(id) {
    setRefusal('cancel', null);
    try {
      await call(taskPath(id), { method: 'DELETE' });
    } catch (refusal) {
      setRefusal('cancel', refusal);
    }
    await loadTasks();
    render();
  }

  // -------------------------------------------------------------- the page

  /**
   * Every section, rebuilt from state — and the keyboard put back where it
   * was. Status is on a timer, so a redraw happens under whoever is using the
   * page; a control that lost focus every four seconds would be unusable with
   * a keyboard, which is the one way this console has to be operable.
   */
  function render() {
    var active = document.activeElement;
    var focused = active && active.id ? active.id : null;
    var caret = null;
    if (focused !== null && active.setSelectionRange && active.type !== 'select-one') {
      try {
        caret = [active.selectionStart, active.selectionEnd];
      } catch (unsupported) {
        caret = null;
      }
    }

    renderStatus();
    renderTasks();
    renderJobTypes();
    renderSettings();
    renderCatalog();
    renderConnect();
    renderService();

    if (focused !== null) {
      var again = document.getElementById(focused);
      if (again !== null) {
        again.focus();
        if (caret !== null && again.setSelectionRange) {
          again.setSelectionRange(caret[0], caret[1]);
        }
      }
    }

    var stamp = document.getElementById('status-stamp');
    stamp.textContent = state.activity
      ? 'read ' + new Date().toLocaleTimeString()
      : '';
  }

  function showGate(refusal) {
    document.getElementById('console').hidden = true;
    var gate = document.getElementById('gate');
    gate.hidden = false;
    var box = document.getElementById('gate-refusal');
    box.textContent = '';
    var built = refusalBox(refusal);
    if (built) {
      box.appendChild(built);
    }
    document.getElementById('bar-name').textContent = 'operator console';
    document.getElementById('bar-state').textContent = '';
    document.getElementById('gate-token').focus();
  }

  function showConsole() {
    document.getElementById('gate').hidden = true;
    document.getElementById('console').hidden = false;
  }

  async function refreshAll() {
    await Promise.all([
      loadSetup(),
      loadInfo(),
      loadActivity(),
      loadCapability(),
      loadSettings(),
      loadCatalog(),
      loadPairingRequests()
    ]);
    await loadTasks();
    state.lastStatusAt = Date.now();
    render();
  }

  async function signIn(token) {
    state.token = token;
    await refreshAll();
    if (state.token === null) {
      return;
    }
    storeToken(token);
    showConsole();
    if (new URLSearchParams(window.location.search).get('section') === 'connect') {
      document.getElementById('panel-connect').scrollIntoView();
    }
    if (state.timer === null) {
      state.timer = window.setInterval(function () {
        if (state.token === null) {
          return;
        }
        state.lastStatusAt = Date.now();
        Promise.all([loadActivity(), loadPairingRequests()]).then(render);
      }, ACTIVITY_MS);
    }
  }

  function start() {
    document.getElementById('gate-form').addEventListener('submit', function (event) {
      event.preventDefault();
      var field = document.getElementById('gate-token');
      var typed = field.value.trim();
      if (typed === '') {
        return;
      }
      field.value = '';
      signIn(typed);
    });
    document.getElementById('status-refresh').addEventListener('click', function () {
      refreshAll();
    });

    var fragment = tokenFromFragment();
    if (fragment !== null) {
      storeToken(fragment);
      signIn(fragment);
      return;
    }
    var stored = readStoredToken();
    if (stored !== null && stored !== '') {
      signIn(stored);
      return;
    }
    showGate(null);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
