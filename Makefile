.PHONY: test preflight

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

preflight:
	python scripts/check_final_unlock.py --protocol configs/protocol_draft.json
