from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import EngineError
from .base import BIND_FAILURE_LINES, PORT_IN_USE, SubprocessEngine, plan_flags, port_in_use_error

ENGINE_NAME = "llama-server"

GRACEFUL_STOP_SECONDS = 30.0

PAGES_ENGINE_FAILED = "pages_engine_failed"

FATAL_SCAN_LINES = 200

FATAL_LINES: tuple[tuple[str, str, str], ...] = (
    (
        "cuda error: out of memory",
        PAGES_ENGINE_FAILED,
        "the card has no room for this model",
    ),
    (
        "out of memory",
        PAGES_ENGINE_FAILED,
        "the card, or this machine's RAM, has no room",
    ),
    (
        "cudart64_",
        PAGES_ENGINE_FAILED,
        "a CUDA runtime DLL is missing: the cudart asset did not unpack",
    ),
    (
        "error while loading shared libraries",
        PAGES_ENGINE_FAILED,
        "a CUDA library llama-server loads from the llm env is missing: "
        "`crucible install llm --force` rebuilds that env",
    ),
    (
        "the code execution cannot proceed",
        PAGES_ENGINE_FAILED,
        "a DLL beside llama-server is missing",
    ),
    (
        "failed to load model",
        PAGES_ENGINE_FAILED,
        "llama.cpp will not read this GGUF",
    ),
    (
        "error loading model",
        PAGES_ENGINE_FAILED,
        "llama.cpp will not read this GGUF",
    ),
    (
        "unknown model architecture",
        PAGES_ENGINE_FAILED,
        "this llama.cpp build does not know this model",
    ),
    *(
        (needle, PORT_IN_USE, "something else on this machine took the port")
        for needle in BIND_FAILURE_LINES
    ),
)


def fatal_reason(line: str) -> tuple[str, str] | None:
    lowered = line.lower()
    for needle, code, reason in FATAL_LINES:
        if needle in lowered:
            return (code, reason)
    return None


class LlamaServerEngine(SubprocessEngine):
    name = ENGINE_NAME

    chat_concurrency = 1
    chat_concurrency_basis = (
        "llama-server is started with --parallel 1 (every llama-server block's "
        "engine_args, on llama-windows and on cuda-linux): one slot generates and "
        "the rest queue inside the server"
    )

    chat_prefill = True
    chat_prefill_basis = (
        "llama-server b10970 takes continue_final_message with add_generation_prompt "
        "false (tools/server/server-common.cpp L1296-1310) and renders the messages "
        "before the final one with the template, then the generation prompt up to "
        "the reasoning start, an empty reasoning block and the final message's "
        "content (common/chat-auto-parser-generator.cpp L45-61)"
    )

    structured_output_formats = frozenset({"json_object", "json_schema"})
    structured_output_fields = frozenset({"json_schema", "grammar"})
    structured_output_basis = (
        "llama-server b10970 reads json_schema and grammar from the body and "
        "response_format json_object and json_schema into the same schema, and answers "
        "any other response_format type with an error "
        "(tools/server/server-common.cpp L1179-1204). It reads no structured_outputs "
        "and no guided_* field; read at the b10970 tag 2026-10-10. On cuda-linux the "
        "door sends every JSON schema to it as an llguidance grammar "
        "(structured.with_llguidance_grammar); on llama-windows the schema goes as sent "
        "and becomes its GBNF grammar (common/json-schema-to-grammar.cpp)"
    )
    json_whitespace_compact = True
    json_whitespace_basis = (
        "Crucible's cuda-linux build of llama-server b10970 has llguidance 1.7.6, the "
        "one the PC's vLLM runs (scripts/build-llama-server-linux.sh, "
        "LLAMA_LLGUIDANCE=ON). The door sends the schema as the grammar "
        "'%llguidance {}\\nstart: %json <schema>', the form llama.cpp writes itself "
        "(common/json-schema-to-grammar.cpp L993-996), which llama-server hands to "
        "llguidance (common/sampling.cpp L213-215): its JSON compiler takes the schema's "
        "x-guidance whitespace_flexible false over its default of true, as vLLM's does. "
        "A grammar the request names is the user's (server-common.cpp L1414-1420, "
        "server-schema.cpp L280), so the generation prompt is not fed through it "
        "(common/common.h L218, sampling.cpp L297) and it holds from the first "
        "generated token, as on vLLM and mlx-lm; read at the b10970 tag 2026-10-10"
    )
    llguidance_grammar_backends = frozenset({"cuda-linux"})
    llguidance_unbuilt_basis = (
        "llama-windows runs ggml-org's own b10970 build, which has no llguidance "
        "(LLAMA_LLGUIDANCE defaults off, CMakeLists.txt L146, and only the 3rd-party "
        "Ubuntu CI job turns it on), so it compiles a JSON schema with its GBNF "
        "converter, whose whitespace rule is fixed: space ::= | \" \" | \"\\n\"{1,2} "
        "[ \\t]{0,20} (common/json-schema-to-grammar.cpp L229, set by every converter at "
        "L816), and no request field changes it (tools/server/server-common.cpp "
        "L1179-1204). A grammar in llguidance's syntax ABORTS that build "
        "(common/sampling.cpp L213-217); read at the b10970 tag 2026-10-10"
    )

    decide_logprobs = True
    max_logprobs = None
    decide_items_basis = (
        "llama-server b10970 answers one prompt per request and checkpoints the "
        "shared state at the last user message, so the items go as one request each"
    )
    decide_basis = (
        "llama-server b10970's /v1/chat/completions maps logprobs/top_logprobs to "
        "n_probs (tools/server/server-common.cpp L1403-1412) and returns "
        "pre-sampling choices[0].logprobs.content[].top_logprobs "
        "(server-task.cpp L282-300, L434-437); n_probs has no cap below the "
        "vocabulary"
    )

    decide_likelihood_route = "forced-tokens"
    decide_likelihood_basis = (
        "llama-server b10970 returns no log-probability for a prompt token "
        "(/v1/completions refuses echo, tools/server/server-common.cpp L1050-1053), "
        "so each candidate is GENERATED as a forced continuation of the context: "
        "/completion with the context's token ids, a GBNF grammar of the "
        "candidate's token ids (<[id]>, src/llama-grammar.cpp parse_token "
        "L186-230) and n_probs with post_sampling_probs false. That logprob is "
        "the raw distribution's: populate_token_probs (tools/server/"
        "server-context.cpp L1964-2020) reads get_token_probabilities "
        "(server-common.cpp L1524-1573), a softmax over llama_get_logits_ith, the "
        "model's logits over the whole vocabulary, with no temperature, while the "
        "grammar and the sampler chain work on the sampler's own copy "
        "(common/sampling.cpp set_logits L130-162, common_sampler_sample "
        "L594-660); asking for pre-sampling probabilities turns backend sampling "
        "off, so the whole vocabulary is there (server-context.cpp L1790-1796). "
        "It is the quantity vLLM's prompt logprobs and the Mac's rows read, at "
        "the same tokens: the door tokenizes through /apply-template and "
        "/tokenize and forces exactly the ids the boundary scored. One request "
        "per candidate in turn (one slot); the context is the same prompt each "
        "time, so cache_prompt reuses it and a recurrent layer restarts from the "
        "checkpoint taken 4 tokens before the prompt's end (server-context.cpp "
        "L3559-3576). Text only. Read at the b10970 tag 2026-10-10"
    )

    sigterm_wait_seconds = GRACEFUL_STOP_SECONDS

    def missing_executable_hint(self) -> str:
        return (
            "llama-server is not installed: run `crucible install llm`, which "
            "pulls the pinned llama.cpp build, and load again"
        )

    @classmethod
    def load_args(
        cls,
        spec: Any,
        weights_dir: Path,
        context: int,
        plan: Any,
        *,
        card_flags: tuple[str, ...] = (),
        source: str = "",
    ) -> list[str]:
        if spec.file is None:
            raise EngineError(
                f"{source}'s {spec.backend} block names no "
                "`file`, and llama-server serves one GGUF. A block for "
                "this backend without a file is a block for nothing"
            )
        mmproj = [] if spec.mmproj is None else ["--mmproj", str(weights_dir / spec.mmproj)]
        return [
            "-m",
            str(weights_dir / spec.file),
            *spec.engine_args,
            *mmproj,
            "-c",
            str(context),
            *plan_flags(plan),
        ]

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        return [
            str(self._python),
            *args,
            "--alias",
            served_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]


    def readiness_description(self) -> str:
        return f"answer {self.base_url}/v1/models with {self._served_name!r}"

    def announced_ready(self) -> str | None:
        fatal = self._fatal_in_log()
        if fatal is not None:
            code, reason, line = fatal
            if code == PORT_IN_USE:
                raise port_in_use_error(
                    self.name, self._port, self._served_name, line
                )
            raise EngineError(
                f"{code}: {self.name} will not come up: {reason}. It said: {line}"
            )
        return super().announced_ready()

    def _fatal_in_log(self) -> tuple[str, str, str] | None:
        for line in self.log_tail(FATAL_SCAN_LINES).splitlines():
            found = fatal_reason(line)
            if found is not None:
                return (found[0], found[1], line.strip())
        return None


__all__ = [
    "ENGINE_NAME",
    "FATAL_LINES",
    "GRACEFUL_STOP_SECONDS",
    "LlamaServerEngine",
    "PAGES_ENGINE_FAILED",
    "PORT_IN_USE",
    "fatal_reason",
]
