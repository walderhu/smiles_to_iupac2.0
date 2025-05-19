
TRASH := __pycache__ __predicted__
FILES := $(shell find . -maxdepth 1 -type f -name "*.py")
DEFAULT_GOAL := c

.PHONY: build clean
.DELETE_ON_ERROR: 

c: 
	@echo "$(FILES)" | xargs -n 1 isort
	@echo "$(FILES)" | xargs -n 1 autopep8 --in-place --aggressive --max-line-length 120
	@echo "$(FILES)" | xargs -n 1 autoflake --remove-all-unused-imports --remove-unused-variables -i
	@echo "Clang format style apply is finished"


%:
	@python3 "_$@.py"

clean:
	@for item in ${TRASH}; do \
		if [ -e "$$item" ]; then \
			read -p "Удалить '$$item'? (y/n): " confirm; \
			if [ "$$confirm" = "y" ] || [ -z "$$confirm" ]; then \
				${RM} -r "$$item"; \
			fi; \
		fi; \
	done
