# Claude Code instructions

Use AGENTS.md as the shared engineering contract. The same requirements apply
to Codex, Claude Code, and all contributors.

The safe sequence is: understand current behavior, add a failing regression test,
make the smallest working change, run real tests, and update the capability matrix.

Do not confuse this project's offline hashing embedding with learned semantic
embeddings; use a real embedding model for semantic production workloads.
If writing any GitHub comment, its entire content must be exactly: @codex review
