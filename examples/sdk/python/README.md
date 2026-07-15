# sdk/python — gemma4 via the generated Python client

```bash
task sdk                                        # once: generate build/sdk/python
cd examples/sdk/python
python3 -m venv .venv && . .venv/bin/activate   # any env manager works
pip install ../../../build/sdk/python
python3 main.py                                 # against a running lem serve (LEM_BASE_URL overrides)
```

The client is the `lem_sdk` package, generated from the OpenAPI spec — typed
request/response models (`choices[].message.{content,thought}`), no
hand-written HTTP.
