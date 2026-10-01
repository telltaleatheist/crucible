'use strict';

(function () {
  var API_HEADER = 'X-Crucible-Api';
  var API_VERSION = '1';
  var CLIENT_HEADER = 'X-Crucible-Client';
  var CLIENT_NAME = 'crucible-playground';
  var TOKEN_KEY = 'crucible.token';
  var TICK_MS = 1000;
  var RECONNECT_MS = 2000;
  var INSTALL_ROUNDS = 5;

  var MEDIA_TYPES = {
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    webp: 'image/webp',
    mp4: 'video/mp4',
    webm: 'video/webm',
    flac: 'audio/flac',
    wav: 'audio/wav',
    mp3: 'audio/mpeg'
  };

  var MEDIA_ORDER = ['image', 'video', 'audio'];

  var ENDED = ['done', 'failed', 'cancelled', 'removed', 'refused'];

  var state = {
    token: null,
    pages: null,
    refusal: null,
    page: null,
    job: null,
    stream: null,
    timer: null
  };

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
    return new Refusal(
      response.status,
      'http_' + response.status,
      'this answer did not come from Crucible: HTTP ' +
        response.status +
        ' with no named refusal in it',
      null
    );
  }

  function headers(extra) {
    var built = { Authorization: 'Bearer ' + state.token };
    built[API_HEADER] = API_VERSION;
    built[CLIENT_HEADER] = CLIENT_NAME;
    if (extra) {
      for (var key in extra) {
        if (Object.prototype.hasOwnProperty.call(extra, key)) {
          built[key] = extra[key];
        }
      }
    }
    return built;
  }

  async function send(path, options) {
    var init = options ? Object.assign({}, options) : {};
    init.headers = headers(init.headers);
    var response;
    try {
      response = await fetch(path, init);
    } catch (unreachable) {
      if (unreachable && unreachable.name === 'AbortError') {
        throw unreachable;
      }
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
    return response;
  }

  async function call(path, options) {
    var response = await send(path, options);
    if (response.status === 204) {
      return null;
    }
    return response.json();
  }

  function jobPath(id) {
    var safe = encodeURIComponent(id);
    return `/v1/jobs/${safe}`;
  }

  function jobEventsPath(id) {
    var safe = encodeURIComponent(id);
    return `/v1/jobs/${safe}/events`;
  }

  function taskPath(id) {
    var safe = encodeURIComponent(id);
    return `/v1/tasks/${safe}`;
  }

  function taskEventsPath(id) {
    var safe = encodeURIComponent(id);
    return `/v1/tasks/${safe}/events`;
  }

  function artifactPath(id, name) {
    var safe = encodeURIComponent(id);
    var file = encodeURIComponent(name);
    return `/v1/jobs/${safe}/artifacts/${file}`;
  }

  function pageLink(page) {
    var query = new URLSearchParams({ type: page.job_type, model: page.id });
    return 'playground.html?' + query.toString();
  }

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
    }
  }

  function forgetToken() {
    try {
      window.localStorage.removeItem(TOKEN_KEY);
    } catch (blocked) {
      return;
    }
  }

  function tokenFromFragment() {
    var hash = window.location.hash;
    if (!hash || hash.length < 2) {
      return null;
    }
    var found = new URLSearchParams(hash.slice(1)).get('token');
    if (!found) {
      return null;
    }
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
    stopTimer();
    showGate(refusal);
  }

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
        } else if (key === 'required') {
          node.required = true;
        } else if (key === 'checked') {
          node.checked = true;
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

  function refusalBox(refusal) {
    if (!refusal) {
      return null;
    }
    return el('div', { class: 'refusal', role: 'status' }, [
      el('code', { class: 'refusal-code', text: refusal.code }),
      el('span', { class: 'refusal-message', text: refusal.message })
    ]);
  }

  var UNITS = ['B', 'kB', 'MB', 'GB', 'TB'];

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

  function downloadText(page) {
    var size = bytesText(page.download_bytes);
    return size === null ? 'downloads on first use' : 'downloads ' + size + ' on first use';
  }

  function secondsText(value) {
    var whole = Math.floor(value);
    var minutes = Math.floor(whole / 60);
    var seconds = whole % 60;
    if (minutes > 0) {
      return minutes + ' min ' + seconds + ' s';
    }
    return seconds + ' s';
  }

  function capitalized(text) {
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  function extensionOf(name) {
    var dot = name.lastIndexOf('.');
    return dot === -1 ? '' : name.slice(dot + 1).toLowerCase();
  }

  function pickArtifact(artifacts, media) {
    for (var index = 0; index < artifacts.length; index += 1) {
      var type = MEDIA_TYPES[extensionOf(artifacts[index])];
      if (type !== undefined && type.indexOf(media + '/') === 0) {
        return artifacts[index];
      }
    }
    return null;
  }

  function choiceValue(field, raw) {
    for (var index = 0; index < field.options.length; index += 1) {
      if (String(field.options[index]) === raw) {
        return field.options[index];
      }
    }
    return raw;
  }

  function paramsOf(fields, form) {
    var params = {};
    for (var index = 0; index < fields.length; index += 1) {
      var field = fields[index];
      var control = form.elements[field.name];
      if (field.kind === 'boolean') {
        params[field.name] = control.checked;
        continue;
      }
      var raw = control.value;
      if (raw.trim() === '') {
        continue;
      }
      if (field.kind === 'integer') {
        params[field.name] = parseInt(raw, 10);
      } else if (field.kind === 'number') {
        params[field.name] = parseFloat(raw);
      } else if (field.kind === 'choice') {
        params[field.name] = choiceValue(field, raw);
      } else {
        params[field.name] = raw;
      }
    }
    return params;
  }

  function selected() {
    var query = new URLSearchParams(window.location.search);
    var type = query.get('type');
    var model = query.get('model');
    if (!type || !model || state.pages === null) {
      return null;
    }
    for (var index = 0; index < state.pages.length; index += 1) {
      var page = state.pages[index];
      if (page.job_type === type && page.id === model) {
        return page;
      }
    }
    return { missing: true, job_type: type, id: model };
  }

  async function loadPages() {
    try {
      state.pages = (await call('/v1/playground')).pages;
      state.refusal = null;
    } catch (refusal) {
      state.pages = null;
      state.refusal = refusal;
    }
  }

  function renderList(body) {
    var pages = state.pages;
    var ready = 0;
    for (var count = 0; count < pages.length; count += 1) {
      if (pages[count].standing === 'ready') {
        ready += 1;
      }
    }
    document.getElementById('models-stamp').textContent =
      ready + ' of ' + pages.length + ' ready now on this server';
    body.appendChild(
      el('p', {
        class: 'lead',
        text:
          'One page per image, video and audio model this server knows. Open one, ' +
          'describe what you want, and press Generate. A model this server does ' +
          'not have yet downloads by itself the first time; one it cannot run ' +
          'says why.'
      })
    );

    var order = [];
    var grouped = {};
    for (var place = 0; place < MEDIA_ORDER.length; place += 1) {
      for (var index = 0; index < pages.length; index += 1) {
        var page = pages[index];
        if (page.media !== MEDIA_ORDER[place]) {
          continue;
        }
        if (grouped[page.makes] === undefined) {
          grouped[page.makes] = [];
          order.push(page.makes);
        }
        grouped[page.makes].push(page);
      }
    }
    if (order.length === 0) {
      body.appendChild(el('p', { class: 'empty', text: 'this build declares no models' }));
      return;
    }

    for (var group = 0; group < order.length; group += 1) {
      var name = order[group];
      var box = el('div', { class: 'rows' });
      box.appendChild(
        el('div', { class: 'row head' }, [
          el('span', { text: capitalized(name) }),
          el('span', { class: 'num', text: String(grouped[name].length) })
        ])
      );
      for (var row = 0; row < grouped[name].length; row += 1) {
        box.appendChild(pageRow(grouped[name][row]));
      }
      body.appendChild(el('div', { class: 'block' }, [box]));
    }
  }

  function pageRow(page) {
    var block = el('div', { class: 'row' });
    block.appendChild(
      el('span', null, [
        el('span', { class: 'row-title', text: page.name }),
        el('span', { class: 'row-id', text: page.id })
      ])
    );
    block.appendChild(el('span', { class: 'chips' }, [chip(page.job_type, 'floor')]));
    block.appendChild(
      el('span', { class: 'row-size' }, [standingChip(page)])
    );
    block.appendChild(
      el('span', { class: 'row-action' }, [
        el('a', {
          class: page.available ? 'button primary' : 'button',
          href: pageLink(page),
          id: 'open-' + page.job_type + '-' + page.id
        }, ['Open'])
      ])
    );
    if (page.standing === 'unavailable') {
      block.appendChild(
        el('div', { class: 'row-span' }, [el('span', { class: 'note', text: page.reason })])
      );
    }
    return block;
  }

  function standingChip(page) {
    if (page.standing === 'ready') {
      return chip('ready', 'ok');
    }
    if (page.standing === 'download') {
      return chip(downloadText(page), 'accent');
    }
    return chip('not available', 'warn');
  }

  function control(field) {
    var id = 'param-' + field.name;
    if (field.kind === 'text') {
      return el('textarea', {
        id: id,
        name: field.name,
        rows: field.required ? 4 : 2,
        placeholder: field.placeholder || null,
        required: field.required,
        spellcheck: 'true'
      });
    }
    if (field.kind === 'boolean') {
      return el('input', {
        id: id,
        name: field.name,
        type: 'checkbox',
        checked: field.default === true
      });
    }
    if (field.kind === 'choice') {
      var select = el('select', { id: id, name: field.name });
      for (var index = 0; index < field.options.length; index += 1) {
        var option = String(field.options[index]);
        select.appendChild(el('option', { value: option, text: option }));
      }
      select.value = String(field.default);
      return select;
    }
    return el('input', {
      id: id,
      name: field.name,
      type: 'number',
      min: field.min === undefined ? null : String(field.min),
      max: field.max === undefined ? null : String(field.max),
      step: field.step === undefined ? 'any' : String(field.step),
      value: field.default === null ? null : String(field.default),
      required: field.required,
      inputmode: field.kind === 'integer' ? 'numeric' : 'decimal'
    });
  }

  function fieldBlock(field) {
    var label = el('label', { class: field.kind === 'text' ? 'field wide' : 'field' }, [
      el('span', { class: 'field-label', text: field.label }),
      control(field)
    ]);
    if (field.hint) {
      label.appendChild(el('span', { class: 'field-hint', text: field.hint }));
    }
    return label;
  }

  function renderPage(body, page) {
    document.getElementById('h-models').textContent = page.missing ? page.id : page.name;
    document.title = (page.missing ? page.id : page.name) + ' — Crucible playground';
    body.appendChild(
      el('p', { class: 'note' }, [el('a', { href: 'playground.html' }, ['All models'])])
    );
    if (page.missing) {
      body.appendChild(
        refusalBox(
          new Refusal(
            404,
            'unknown_model',
            'this server has no ' + page.job_type + ' model ' + page.id +
              '; the list of every model it knows is one link back'
          )
        )
      );
      return;
    }
    document.getElementById('models-stamp').textContent =
      page.job_type + ' · makes ' + page.makes;
    if (page.standing === 'download') {
      body.appendChild(
        el('p', { class: 'note' }, [
          standingChip(page),
          ' ',
          page.reason + '. You can press Generate now.'
        ])
      );
    }
    if (!page.available) {
      body.appendChild(
        el('div', { class: 'refusal', role: 'status' }, [
          el('code', { class: 'refusal-code', text: 'not_ready' }),
          el('span', { class: 'refusal-message', text: page.reason })
        ])
      );
      return;
    }

    var form = el('form', { id: 'generate-form', class: 'generate' });
    var wide = el('div', { class: 'fields' });
    var narrow = el('div', { class: 'fields' });
    for (var index = 0; index < page.fields.length; index += 1) {
      var field = page.fields[index];
      (field.kind === 'text' ? wide : narrow).appendChild(fieldBlock(field));
    }
    form.appendChild(wide);
    form.appendChild(narrow);
    form.appendChild(
      el('div', { class: 'controls' }, [
        el('button', { id: 'generate', class: 'button primary', type: 'submit' }, [
          'Generate'
        ]),
        el('button', {
          id: 'cancel',
          class: 'button',
          type: 'button',
          hidden: true,
          onclick: function () {
            cancel();
          }
        }, ['Cancel'])
      ])
    );
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      if (!form.reportValidity()) {
        return;
      }
      generate(page, paramsOf(page.fields, form));
    });
    body.appendChild(form);
    body.appendChild(el('div', { id: 'job-body', class: 'job' }));
    renderJob();
  }

  function render() {
    var body = document.getElementById('models-body');
    body.textContent = '';
    document.getElementById('models-stamp').textContent = '';
    var refusal = refusalBox(state.refusal);
    if (refusal) {
      body.appendChild(refusal);
      return;
    }
    if (state.pages === null) {
      body.appendChild(el('p', { class: 'empty', text: 'reading…' }));
      return;
    }
    state.page = selected();
    if (state.page === null) {
      renderList(body);
    } else {
      renderPage(body, state.page);
    }
  }

  function isActive(job) {
    return job !== null && ENDED.indexOf(job.status) === -1;
  }

  function statusText(job) {
    var waited = secondsText((Date.now() - job.since) / 1000);
    if (job.status === 'submitting') {
      return 'Sending the job…';
    }
    if (job.status === 'installing') {
      return installText(job.install) + ' (' + waited + ')';
    }
    if (job.status === 'queued') {
      var where = job.position === null ? 'in line' : 'number ' + job.position +
        (job.of ? ' of ' + job.of : '') + ' in line';
      return 'Waiting for the server, ' + where + ' (' + waited + ')';
    }
    if (job.status === 'running') {
      var said = job.message ? capitalized(job.message) : 'Working';
      var share = job.fraction === null ? '' : ', ' + Math.round(job.fraction * 100) + '%';
      return said + share + ' (' + waited + ')';
    }
    if (job.status === 'fetching') {
      return 'Fetching the result…';
    }
    if (job.status === 'done') {
      return 'Done in ' + secondsText((job.ended - job.since) / 1000) +
        (job.seed === null ? '' : ' · seed ' + job.seed);
    }
    if (job.status === 'cancelled') {
      return 'Cancelled.';
    }
    if (job.status === 'removed') {
      return 'It left the queue without running' +
        (job.message ? ': ' + job.message : '') + '. Press Generate to send it again.';
    }
    return '';
  }

  function installText(install) {
    var said = install.ours
      ? 'Downloading what this model needs first, once'
      : 'Waiting for another install on the server to finish';
    if (install.step) {
      said += ': step ' + install.step.index + ' of ' + install.step.total + ', ' +
        install.step.name;
    }
    var done = bytesText(install.bytesDone);
    if (done !== null) {
      var total = bytesText(install.bytesTotal);
      said += ', ' + done + (total === null ? ' so far' : ' of ' + total);
    }
    return said + '. The job starts by itself after it';
  }

  function resultView(job) {
    var result = job.result;
    var media;
    if (state.page.media === 'image') {
      media = el('img', { src: result.url, alt: 'the generated image', class: 'result-media' });
    } else if (state.page.media === 'video') {
      media = el('video', { src: result.url, controls: '', class: 'result-media' });
    } else {
      media = el('audio', { src: result.url, controls: '', class: 'result-audio' });
    }
    return el('div', { class: 'result' }, [
      media,
      el('p', { class: 'note' }, [
        el('a', { href: result.url, download: result.file, id: 'download' }, [
          'Download ' + result.file
        ])
      ])
    ]);
  }

  function renderJob() {
    var box = document.getElementById('job-body');
    if (box === null) {
      return;
    }
    box.textContent = '';
    var job = state.job;
    var active = isActive(job);
    var generate = document.getElementById('generate');
    var cancelButton = document.getElementById('cancel');
    if (generate !== null) {
      generate.disabled = active;
    }
    if (cancelButton !== null) {
      cancelButton.hidden = !cancellable(job);
    }
    if (job === null) {
      return;
    }
    var text = statusText(job);
    if (text) {
      var line = el('div', { class: 'progress-line', text: text, role: 'status' });
      if (active) {
        var install = job.status === 'installing' ? job.install : null;
        var known = install !== null
          ? Boolean(install.bytesTotal) && install.bytesDone !== null
          : job.status === 'running' && job.fraction !== null;
        var bar = el('div', {
          class: known ? 'bar' : 'bar indeterminate',
          role: 'progressbar',
          'aria-valuetext': text
        }, [el('span')]);
        if (known) {
          var share = install !== null ? install.bytesDone / install.bytesTotal : job.fraction;
          bar.firstChild.style.width = (share * 100).toFixed(1) + '%';
        }
        var detail = install !== null && install.line
          ? el('div', { class: 'field-hint', text: install.line })
          : null;
        box.appendChild(el('div', { class: 'progress' }, [line, bar, detail]));
      } else {
        box.appendChild(el('div', { class: 'progress' }, [line]));
      }
    }
    var refusal = refusalBox(job.refusal);
    if (refusal) {
      box.appendChild(refusal);
    }
    if (job.result) {
      box.appendChild(resultView(job));
    }
  }

  function stopStream() {
    if (state.stream !== null) {
      state.stream.abort();
      state.stream = null;
    }
  }

  function stopTimer() {
    if (state.timer !== null) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
  }

  function forgetResult() {
    if (state.job && state.job.result) {
      URL.revokeObjectURL(state.job.result.url);
    }
  }

  function newJob() {
    return {
      id: null,
      status: 'submitting',
      position: null,
      of: null,
      fraction: null,
      message: null,
      seed: null,
      install: null,
      refusal: null,
      result: null,
      since: Date.now(),
      ended: null
    };
  }

  function newInstall(details) {
    var step = details.step || null;
    return {
      taskId: details.task_id,
      ours: details.reason !== 'task_busy',
      message: details.message || null,
      step: step && step.name ? step : null,
      bytesDone: null,
      bytesTotal: null,
      line: details.line || null,
      ended: null
    };
  }

  function finish(job, status) {
    job.status = status;
    job.ended = Date.now();
    stopTimer();
    renderJob();
  }

  function pause(ms) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, ms);
    });
  }

  async function generate(page, params) {
    stopStream();
    stopTimer();
    forgetResult();
    var job = newJob();
    state.job = job;
    renderJob();
    state.timer = window.setInterval(renderJob, TICK_MS);
    var receipt = null;
    for (var round = 0; receipt === null; round += 1) {
      try {
        receipt = await call('/v1/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: page.job_type,
            model: page.id,
            params: params,
            queue: {}
          })
        });
      } catch (refusal) {
        if (state.job !== job) {
          return;
        }
        if (!isInstalling(refusal) || round >= INSTALL_ROUNDS) {
          job.refusal = refusal;
          finish(job, 'refused');
          return;
        }
        job.status = 'installing';
        job.install = newInstall(refusal.details);
        renderJob();
        var outcome = await followInstall(job, job.install);
        if (state.job !== job) {
          return;
        }
        if (outcome === 'cancelled') {
          finish(job, 'cancelled');
          return;
        }
        if (outcome !== 'done') {
          job.refusal = outcome;
          finish(job, 'failed');
          return;
        }
        job.status = 'submitting';
        job.install = null;
        renderJob();
      }
    }
    job.id = receipt.job_id;
    if (receipt.queued) {
      job.status = 'queued';
      job.position = receipt.position === undefined ? null : receipt.position;
    } else {
      job.status = 'running';
    }
    renderJob();
    follow(job, page);
  }

  function isInstalling(refusal) {
    return (
      refusal instanceof Refusal &&
      refusal.code === 'installing' &&
      Boolean(refusal.details) &&
      typeof refusal.details.task_id === 'string'
    );
  }

  async function followInstall(job, install) {
    var cursor = { last: 0 };
    while (state.job === job) {
      try {
        await readEvents(taskEventsPath(install.taskId), cursor, function (name, data) {
          applyInstallEvent(install, name, data);
          renderJob();
        });
      } catch (refusal) {
        if (refusal instanceof Refusal && refusal.status !== 0) {
          return refusal;
        }
      }
      if (install.ended !== null) {
        return install.ended;
      }
      await pause(RECONNECT_MS);
    }
    return null;
  }

  function applyInstallEvent(install, name, data) {
    if (name === 'step') {
      install.step = data;
      install.bytesDone = null;
      install.bytesTotal = null;
      install.line = null;
    } else if (name === 'progress') {
      if (data.line !== undefined) {
        install.line = data.line;
      } else {
        install.bytesDone = data.bytes_done;
        install.bytesTotal = data.bytes_total;
      }
    } else if (name === 'done') {
      install.ended = 'done';
    } else if (name === 'failed') {
      install.ended = new Refusal(
        0,
        data.code || 'install_failed',
        'installing what this model needs failed: ' +
          (data.message || 'task ' + install.taskId + ' failed') +
          '. Press Generate to try again',
        null
      );
    } else if (name === 'cancelled') {
      install.ended = 'cancelled';
    }
  }

  async function follow(job, page) {
    var cursor = { last: 0 };
    while (state.job === job && isActive(job) && job.status !== 'fetching') {
      try {
        await readEvents(jobEventsPath(job.id), cursor, function (name, data) {
          if (state.job === job) {
            applyEvent(job, page, name, data);
            renderJob();
          }
        });
      } catch (refusal) {
        if (refusal instanceof Refusal && refusal.status !== 0) {
          job.refusal = refusal;
          finish(job, 'failed');
          return;
        }
      }
      if (state.job !== job || !isActive(job) || job.status === 'fetching') {
        return;
      }
      await pause(RECONNECT_MS);
    }
  }

  async function readEvents(path, cursor, onEvent) {
    var controller = new AbortController();
    state.stream = controller;
    var extra = { Accept: 'text/event-stream' };
    if (cursor.last > 0) {
      extra['Last-Event-ID'] = String(cursor.last);
    }
    var response = await send(path, { headers: extra, signal: controller.signal });
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
        return;
      }
      buffer += decoder.decode(chunk.value, { stream: true });
      var cut = buffer.indexOf('\n\n');
      while (cut !== -1) {
        var event = parseFrame(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
        cut = buffer.indexOf('\n\n');
        if (event === null) {
          continue;
        }
        if (event.id !== null) {
          cursor.last = event.id;
        }
        onEvent(event.name, event.data);
      }
    }
  }

  function parseFrame(frame) {
    var name = null;
    var payload = null;
    var id = null;
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
      } else if (field === 'id') {
        id = parseInt(value, 10);
      } else if (field === 'data') {
        try {
          payload = JSON.parse(value);
        } catch (notJson) {
          payload = null;
        }
      }
    }
    if (name === null) {
      return null;
    }
    return {
      name: name,
      id: id === null || isNaN(id) ? null : id,
      data: payload === null ? {} : payload
    };
  }

  function applyEvent(job, page, name, data) {
    if (name === 'queued') {
      job.status = 'queued';
      job.position = data.position === undefined ? null : data.position;
      job.of = data.of === undefined ? null : data.of;
    } else if (name === 'started') {
      job.status = 'running';
    } else if (name === 'progress') {
      job.status = 'running';
      job.fraction = typeof data.fraction === 'number' ? data.fraction : null;
      if (data.message) {
        job.message = data.message;
      }
    } else if (name === 'done') {
      if (typeof data.seed === 'number') {
        job.seed = data.seed;
      }
      job.status = 'fetching';
      fetchResult(job, page, data.artifacts || []);
    } else if (name === 'failed') {
      var error = data.error || {};
      job.refusal = new Refusal(0, error.code || 'failed', error.message || 'the job failed', null);
      finish(job, 'failed');
    } else if (name === 'cancelled') {
      finish(job, 'cancelled');
    } else if (name === 'removed') {
      job.message = data.message || data.reason || null;
      finish(job, 'removed');
    }
  }

  async function fetchResult(job, page, artifacts) {
    var name = pickArtifact(artifacts, page.media);
    if (name === null) {
      job.refusal = new Refusal(
        0,
        'no_result',
        'the job finished with ' + JSON.stringify(artifacts) + ' and none of them is ' +
          'a file this page can show as ' + page.media,
        null
      );
      finish(job, 'failed');
      return;
    }
    var blob;
    try {
      var response = await send(artifactPath(job.id, name));
      blob = await response.blob();
    } catch (refusal) {
      job.refusal = refusal instanceof Refusal
        ? refusal
        : new Refusal(0, 'unreachable', String(refusal), null);
      finish(job, 'failed');
      return;
    }
    if (state.job !== job) {
      return;
    }
    var extension = extensionOf(name);
    var typed = new Blob([blob], { type: MEDIA_TYPES[extension] });
    job.result = {
      url: URL.createObjectURL(typed),
      file: page.id + '-' + job.id.slice(0, 8) + '.' + extension
    };
    finish(job, 'done');
  }

  function cancellable(job) {
    if (!isActive(job)) {
      return false;
    }
    if (job.status === 'installing') {
      return job.install !== null && job.install.ours;
    }
    return job.id !== null;
  }

  async function cancel() {
    var job = state.job;
    if (!cancellable(job)) {
      return;
    }
    var path = job.status === 'installing' ? taskPath(job.install.taskId) : jobPath(job.id);
    try {
      await call(path, { method: 'DELETE' });
    } catch (refusal) {
      job.refusal = refusal;
      renderJob();
    }
  }

  function showGate(refusal) {
    document.getElementById('playground').hidden = true;
    var gate = document.getElementById('gate');
    gate.hidden = false;
    var box = document.getElementById('gate-refusal');
    box.textContent = '';
    var built = refusalBox(refusal);
    if (built) {
      box.appendChild(built);
    }
    document.getElementById('gate-token').focus();
  }

  function showPlayground() {
    document.getElementById('gate').hidden = true;
    document.getElementById('playground').hidden = false;
  }

  async function signIn(token) {
    state.token = token;
    await loadPages();
    if (state.token === null) {
      return;
    }
    storeToken(token);
    showPlayground();
    render();
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
