# The intent of Crucible

Product direction recorded from Owen's instructions on 2026-09-16.

This is the canonical statement of what Crucible is meant to be. Read it before
changing installation, model ownership, routing, or integration with BookForge
and Foundry. It governs product intent when older phase plans disagree. It is an
acceptance contract, not a claim that every requirement is already implemented
or verified. Release reports record what actually passed.

## The experience

A person with no technical background installs BookForge or Foundry, completes
the app's setup, and uses it. Crucible feels like a natural extension of that app.
They do not need to learn Python, environments, model caches, ports, bearer tokens,
or an inference engine's command line. They normally never open Crucible's UI.

The app explains meaningful choices and shows installation, download and repair
progress. Crucible performs the work through its API. The same controls should
work when Crucible is on another computer. Its own UI and tray remain useful for
maintenance, diagnostics, service control and optional connection approval.

### The Ollama standard (Owen, 2026-09-16)

Crucible should feel like Ollama: installed once, then set and forget. Judge any
proposed change against these.

- **Configuration happens in BookForge and Foundry.** Their own settings and
  setup pages pass configuration through to Crucible programmatically. A user
  may open Crucible's own UI occasionally for a minor adjustment; needing it to
  complete a normal workflow is a defect, not a documentation problem.
- **Crucible manages the models on the user's behalf**, as Ollama does. The app
  declares what it needs; Crucible obtains, stores, serves and removes it.
- **A local server is FOUND, not configured.** An app looks at the standard port,
  sees a Crucible answering and connects — no token typed and no file hunted, the
  way any program finds Ollama.
- **Another machine reaches it the way it reaches Ollama.** Ollama binds loopback
  by default and the operator widens it (`OLLAMA_HOST=0.0.0.0:11434`); what then
  protects the open port is the NETWORK — a firewall, a VPN, a tailnet — and not a
  credential every caller must carry. Crucible follows that. The Mac using the
  PC's engine is a first-class case, not an advanced one, and the bind address
  plus the operator's own network controls are what decide who may reach it. The
  `host`/`advertise` split Crucible already has is the mechanism; what changes is
  that reaching it stops being gated on a token.

  Stated plainly because it is the trade being made: an engine bound to `0.0.0.0`
  on an untrusted network is usable by anyone who can route to it. That is
  precisely Ollama's posture, and precisely why its own documentation tells
  people to put it behind something. The default stays loopback for that reason.
- **If it is missing, the app installs it.** BookForge or Foundry finding no local
  Crucible installs one rather than sending the user elsewhere to get it.
- **The tray icon and the installer are the whole of the user's direct contact.**
  Open it, close it, install it, remove it, at will.
- **Windows alone runs everything the apps need.** Installing into WSL unlocks
  SGLang/vLLM and the Python job types, so there is an incentive to but never a
  requirement; the Windows service is then the pass-through to that guest.

### Any Crucible that answers works (Owen, 2026-09-24)

> *"lets modify bookforge and foundry so they dont require any particular crucible
> server. if it can make the call to the crucible server then it should work."*

Part of the Ollama standard: nobody checks which Ollama they have before they use
it. This supersedes the lockstep ruling of 2026-09-20, under which the SDK demanded
every field it knew and an app one release ahead of its server refused it by name —
which, after the 1.0.24 repin, failed ten BookForge keepers over fields their fake
servers simply did not have yet, and let one new informational voice field break
every `info()` probe.

- **Informational fields are tolerant.** A field that informs a display, or a
  number a caller may use — an estimate, a note, a count, a timestamp, a pin, a
  measurement — reads as `null` when a server does not state it. Version skew is
  weather, not misconfiguration.
- **Load-bearing fields are not.** What a call cannot be done right without — an
  id, a job's status, the artifacts to fetch, the audio, a chat's content, an
  answer, an error's code, anything the client branches on — is refused by name
  when missing.
- **Load-bearing for ONE path is not load-bearing for the read.** A job's
  `chunks_done` is what a resume differences against, but every other caller of
  `job()` only needs the status. It reads as `null` ("unknown", never "none
  done") and the resume decides what to do with a server that cannot say
  (bookforge-pc-1 found it while repinning Foundry, 2026-09-24).
- **A present field of the wrong type is still refused.** API v1 adds fields and
  never retypes them, so a wrong type is a broken server, not an old one.
- **Unknown event kinds are carried, not refused**, and one unreadable voice or
  model row in `info()` is set aside with its reason rather than failing the probe.
- **Misconfiguration is still refused by name**: nothing answering, something that
  is not a Crucible, a wrong token, a different API major, a pairing version the
  app does not speak. There is no minimum-version check beyond the API major.

`sdk/ts/src/shape.ts` states the rule where the readers are; `sdk/ts/README.md`
("Any Crucible that answers") states it for an app.

### An outlet for the whole app suite (Owen, 2026-09-23)

> *"crucible is supposed to operate like ollama but with my custom options. it's an
> outlet for other apps to plug into and use basically. it provides access to all
> actions my app suite would need. transcription, translation, etc. ollama doesnt
> have access to sglang or vllm. crucible does. thats the idea"*

> *"crucible is agnostic about the task thats being run. it has specialized tasks,
> like translate/simplify/etc, but over anything else, it's a gpu orchestrator. if
> tasks require special handling, it can add a unique verb for the task, but lean
> toward managing orchestration. the apps that use it can handle everything
> surrounding that."*

BookForge and Foundry are the first clients, not the only ones: Briefcase and
ContentStudio plug in the same way. Every new request is judged against this test.

- **Express it through a door that already exists.** Chat, `/v1/decide` (options and
  text in, a distribution out), and the job types (`asr`, `tts`, `denoise`, …).
  Owen's own example: finding interesting sections of a book or video is options
  plus the text to match them to, answered back to the calling app. It is not an
  analysis feature inside the server.
- **A unique verb only for special handling:** a different forward pass, engine or
  model family. Prompts, chunking, digests, thresholds and workflow stay in the app.
- **Knobs are the engine's knobs, not the app's tasks.** When two proposed acts
  differ only in a parameter, they become one act with that parameter. Owen:
  *"if the only difference is the context limit then make it one class and give it
  the ability to set the context limit"*. That gave one `generate` class, an 8k
  default, and a context the client may request up to a ceiling per host and per
  model, refused by name above it.
- **CPU-only work stays in the app** (2026-09-17): *"if its cpu only, that means it
  stays local"*. Crucible orchestrates the accelerator.

Idiot-proof is the acceptance bar. A step that a non-technical person would not
think to take is a step that does not exist.

## Who owns what

| Concern | Owner |
|---|---|
| Books, documents, projects and workflow | BookForge or Foundry |
| Which capabilities and models the app needs | The app, informed by the user's choices and server capabilities |
| Normal settings and setup interface | The app |
| Downloading, verifying, storing and removing managed model files | Crucible |
| Backend-compatible model format and runtime preparation | Crucible, satisfying the app's requested capability |
| Loading models, inference, device admission and resource leases | Crucible |
| Application work queues and choosing the next job | The app |
| Native/WSL service lifecycle and migration | Crucible's Windows controller |
| Provider connections and inference routes | Configured through the app; executed and managed by Crucible |

BookForge chooses its voice models and other required capabilities. Foundry
chooses its input-processing and language models: currently document reading/OCR,
cleanup, translation and analysis. Transcription, wherever an app offers it,
follows the same ownership rule. This contract does not itself add a new app workflow. A bare
Crucible installation does not choose an app's model collection or download those
weights. The distinction between installing a runtime and downloading a model
must remain explicit in code and progress reporting.

## Installation and first use

1. Installing either app includes a guided setup that finds a compatible existing
   local Crucible or installs it from the published release. End users do not need
   a GitHub token, the GitHub CLI, Python or a developer checkout.
2. Provider and model choices happen before expensive model downloads. Choosing
   an existing Ollama or cloud route must not first download a redundant local LLM.
3. The app asks Crucible to prepare the selected/default capabilities during setup.
   This is part of completing app installation, not a separate manual Crucible task.
4. Progress and failures are visible in the app. Completion means the selected
   capabilities are actually ready; downloading a file alone does not prove that.
5. Installing the other app reuses the same local Crucible and suitable existing
   models. It adds only what its selected workflows require.

The current packaging decision is that the Windows app installer installs the app,
and its first-launch wizard performs service and capability setup. This can satisfy
the intended experience without duplicating orchestration inside NSIS. It must
remain automatic after the user's necessary choices, including across a restart.
OS-required elevation or reboot is explained and resumed; it cannot be wished away.

## Windows works without WSL

Native Windows is a supported, useful installation on its own. A user who cannot
or will not install WSL must be able to use Foundry's existing document workflows,
including page reading/OCR and its language-model operations. The presence of a
WSL guest must not be a prerequisite for the Windows service to start or function.

The service reports which capabilities the actual backend supports. BookForge
and Foundry prepare compatible capabilities rather than requesting Linux-only
runtimes on Windows. A capability that needs a different backend must be explained
in the app; it must not silently become a broken or degraded route.

WSL is an optional managed engine upgrade. The Windows controller guides OS setup,
installs its Crucible guest and compatible runtimes, carries configuration across,
and connects the apps to the active engine. Apps keep one connection for that
machine rather than exposing two independently configured services to the user.
Explicit stop intent survives recovery checks. Closing a tray window does not
silently kill work.

## One managed copy of a model

BookForge, Foundry, native Windows and WSL must not accumulate independent copies
of the same required model. There is one managed inventory per machine, with a
clear owner and active storage location. Both apps reuse suitable existing weights.

The invariant is about duplicate model payloads, not forcing every backend to read
one literal directory. A Windows GGUF and a Linux SGLang/vLLM checkpoint can be
different representations of the same model. Compatibility is checked by model
identity, revision, format and backend requirements; a name such as "27B" alone
does not establish that two downloads are interchangeable.

When moving to WSL, Crucible prepares and verifies all required destination models,
activates and authenticates the guest, then removes obsolete native weights using
their owning catalog. Temporary overlap during a recoverable migration is allowed;
permanent redundant copies after a successful migration are not. Interrupted cleanup
must be recorded and resumable. Never remove the last usable source before the
destination is ready. Runtime binaries are distinct from model weights.

The apps declare demand; Crucible manages physical storage. Removing one app or
changing one workflow must not delete a model still required by the other app.
Returning to native execution may require preparing its model format again, with
that cost and progress made visible. Do not keep every format indefinitely just in
case it might be needed later.

## Existing providers and cloud models

Crucible can satisfy appropriate requests through its managed local engines,
another Crucible server, an existing Ollama instance, OpenAI, or Anthropic/Claude.
The app configures the selected provider, model and credential programmatically.
The user should not have to open Crucible to repeat those choices.

If Ollama already serves the desired model, route the request to Ollama rather
than downloading an identical model into Crucible. Ollama retains ownership of its
files. Crucible must not delete another provider's model store to reconcile its own
inventory. A cloud route likewise requires no local copy of that provider's model.

Different providers expose different capabilities. Route only supported operations
and report incompatibility explicitly. Choosing a local route must not silently
send data to a paid cloud provider, or vice versa. Credentials belong in trusted
configuration paths, not renderer state or diagnostic output.

## Connections should be easy

An app on the same machine discovers and adopts the existing local installation
and connection without making the user find a token file. Remote setup accepts a
bare IP address or hostname. The current connection contract uses the standard
port 7100 when omitted and permits an explicit custom port; it does not scan every
port on the host.

Discovery identifies a server; authorization grants access. Remote setup uses a
short matching-code approval through an already trusted BookForge or Foundry, with
Crucible's maintenance UI available as another approval surface. A server must not
publish its bearer token merely because a stranger knows its IP address. After
pairing, routine authenticated configuration and model management happen in the app.

Local use works without configuring network interfaces. Enabling access from other
devices is an app-level choice that Crucible applies consistently to its bind,
advertised address and owned networking configuration. The address offered to a
remote app must actually reach the active engine; neither a loopback address nor
an internal WSL address is a usable remote connection merely because it works on
the server itself. Report connection failures in the app with an actionable cause.

## Acceptance scenarios

The following are the practical definition of done. Record the platform, release,
selected models/provider and observed result for each; mocks alone do not establish
installation or GPU support.

- Fresh Windows, no WSL: install Foundry, complete setup and perform real document
  reading and language-model operations through native Crucible.
- Fresh Windows: install BookForge, choose its supported capabilities, complete
  setup and render with the selected voice route.
- Install the second app: it reuses the existing service and shared required models.
- Existing Ollama: choose its model during setup and complete a request without
  downloading an equivalent Crucible model.
- Cloud configuration: configure OpenAI or Claude through the app and complete a
  supported request without a local model download.
- Optional WSL: perform the guided move, verify inference through the guest, and
  verify obsolete native weights are gone while the Windows controller still works.
- Interrupted WSL move: retain a usable source before activation; after activation,
  resume cleanup safely without deleting the guest's models or unrelated files.
- Remote server: enter an IP, approve pairing and configure/use capabilities from
  the app without manually copying a token.
- Upgrade, stop/start, reboot and preserving-data uninstall/reinstall: retain the
  intended configuration/data, respect stop intent, and restore a working connection.

## Keeping this document useful

Update this document when Owen changes the product direction. Implementation plans
must link here and identify their gaps against it. Keep temporary release failures,
training/GPU availability and signing credentials out of the intent document; put
those in the dated work log and release validation report.

Related contracts: [architecture](ARCHITECTURE.md),
[local lifecycle](LOCAL-LIFECYCLE.md), [installation](INSTALL-UNINSTALL.md), and
[historical build plan](PLAN.md).
