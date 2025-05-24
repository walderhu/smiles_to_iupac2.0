TRASH := __pycache__ __predicted__
FILES := $(shell find . -maxdepth 1 -type f -name "*.py")
DEFAULT_GOAL := c

.PHONY: build clean
.DELETE_ON_ERROR: 

c:
	@echo "Applying strict code formatting..."
	@echo "$(FILES)" | xargs -n 1 isort --profile=black --line-length=120 --force-single-line-imports --force-sort-within-sections
	@echo "$(FILES)" | xargs -n 1 autoflake \
		--remove-all-unused-imports \
		--remove-unused-variables \
		--remove-duplicate-keys \
		--expand-star-imports \
		-i
	@echo "$(FILES)" | xargs -n 1 autopep8 --in-place --aggressive --aggressive --max-line-length=120 --select=E,W,C --ignore=E203,E501
	@echo "$(FILES)" | xargs -n 1 black --line-length 120 --skip-string-normalization -q
	@echo "$(FILES)" | xargs -n 1 flake8 --max-line-length=120 --ignore=E203,E266,E501,W503 --max-complexity=10 || \
		(echo "Flake8 found issues that need manual attention"; exit 0)
	@echo "Strict code formatting complete"

clean:
	@for item in ${TRASH}; do \
		if [ -e "$$item" ]; then \
			read -p "Удалить '$$item'? (y/n): " confirm; \
			if [ "$$confirm" = "y" ] || [ -z "$$confirm" ]; then \
				${RM} -r "$$item"; \
			fi; \
		fi; \
	done

# pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
# pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
# 