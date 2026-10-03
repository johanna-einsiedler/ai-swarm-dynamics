# Validation sheets

Random samples (seed 42, agent-authored messages) of extracted rows, for
hand-checking. Regenerate with `python scripts/make_validation.py`. Fill the
empty columns with 1 or 0; precision is reported in the top-level README.

- `mentions.csv`: 50 mention edges. `target_correct` = 1 if the alias refers to
  `resolved_target`; `is_addressed_to_target` = 1 if the message speaks *to*
  that agent rather than *about* it.
- `questions.csv`: 50 extracted questions with the sentence before and after.
  `is_real_question` = 1 if it is a question at all;
  `expects_answer_from_another_agent` = 1 if it asks another agent for
  something (not rhetorical, not addressed to a human).
- `overlaps.csv`: 50 overlap links with at least 40 shared characters.
  `is_copied_from_source` = 1 if the message took the shared text from the
  linked source message (quote, forwarded link, repeated instruction).
