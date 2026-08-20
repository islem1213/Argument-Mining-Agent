# Processed data

Store dataset-specific train, validation, and test conversions here; retain the conversion script and split identifiers.

The component trainer expects JSONL rows shaped as:

```json
{"text": "Later starts improve learning.", "label": "claim"}
```

Valid labels are `major_claim`, `claim`, `premise`, and `non_argumentative`. Keep PE/AAEC licensing and original split metadata alongside the conversion script.
