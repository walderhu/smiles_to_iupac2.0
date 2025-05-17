
TRASH := __pycache__ __predicted__
FILES := $(shell find . -maxdepth 1 -type f -name "*.py")

.PHONY: build clean
.DELETE_ON_ERROR: 

all:
	@python3 "/home/lipatovdn/smiles_to_iupac2.0/model/_train🚀.py"

%:
	@python3 "_$@.py"

install:
	@source install.sh

clean:
	@for item in ${TRASH}; do \
		if [ -e "$$item" ]; then \
			read -p "Удалить '$$item'? (y/n): " confirm; \
			if [ "$$confirm" = "y" ] || [ -z "$$confirm" ]; then \
				${RM} -r "$$item"; \
			fi; \
		fi; \
	done

c: 
	@echo "$(FILES)" | xargs -n 1 isort
	@echo "$(FILES)" | xargs -n 1 autopep8 --in-place --aggressive --max-line-length 120
	@echo "$(FILES)" | xargs -n 1 autoflake --remove-all-unused-imports --remove-unused-variables -i
	@echo "Clang format style apply is finished"
