PROJECT_ROOT = $(CURDIR)
PYTHON = python

.PHONY:
install: requirements.txt
	$(PYTHON) -m pip install -r requirements.txt --upgrade

.PHONY:
test: tests testing conv attentions
	$(PYTHON) -m unittest discover -s $(PROJECT_ROOT)/tests -t $(PROJECT_ROOT)/tests