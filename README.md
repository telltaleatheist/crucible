# Crucible

An inference server for Owen's apps. One server binary per backend, one API, any client.

BookForge, Foundry, Content Studio and Briefcase all need the same class of hardware: a
GPU with a language model, a vision-language model, a TTS model, an aligner or a voice
converter resident on it. Crucible is the thing that gets hot. It runs the models. It
never knows what an audiobook or a cleanup pass is.

See `docs/DESIGN.md` for the architecture and `docs/PLAN.md` for the build order.
