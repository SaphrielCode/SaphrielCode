Published binary results are byte-exact4MiB parts, streamed through the authenticated GitHub app.
Run .venv-data\Scripts\python.exe tools\restore_published_artifacts.py --all to restore checkpoints and results/ExperimentArtifacts.zip.
Use python -m zipfile -e results/ExperimentArtifacts.zip . to unpack raw reports/logs.
RunChat.cmd restores only the selected inference weights and tokenizer automatically.
Every part and reassembled file is SHA-256 checked; existing different files are never overwritten.
Dataset source corpora remain local; source URLs, hashes, licenses and selected synthetic data are included in the results archive.

Stage 1 adaptation artifacts include model_assets/stage-one.pt for chat, the final optimizer recovery checkpoint, and results/StageOne_Artifacts.zip containing frozen Stage1 examples, telemetry, and evaluations.
Restore with tools/restore_published_artifacts.py --all and extract StageOne_Artifacts.zip at the project root to recover the training data and raw run records.
The deployed checkpoint is selected by validation loss at minute30. The full four-hour final state is retained. Generated prose can still miss facts and change names.
