package/exciton/build: ## DEV Build
	@cd modules/exciton-package-nlp && rm -rf exciton.egg-info && python3 setup.py sdist bdist_wheel

package/exciton/build_and_install: ## DEV build and install
	@cd modules/exciton-package-nlp && rm -rf exciton.egg-info && python3 setup.py sdist bdist_wheel && pip install -U .

package/exciton/release: guard-V ## Release the package
	@cd modules/exciton-package-nlp && bump2version $(V) && git push --tags
	@cd modules/exciton-package-nlp && rm -r exciton.egg-info && rm -r dist && python3 setup.py sdist bdist_wheel
	@cd modules/exciton-package-nlp/dist && twine upload *.whl

package/exciton/docs: ## Build documentation
	@cd modules/exciton-package-nlp/docs && make clean && (rm -r source || true ) && make clean html
	@cd modules/exciton-package-nlp/docs && sphinx-apidoc --force -o source ../exciton/
	@cd modules/exciton-package-nlp/docs && make html
	@(rm -r /var/www/html/exciton || true) && cd modules/exciton-package-nlp/docs/_build && cp -r html /var/www/html/exciton

package/exciton/models2s3: ## Sync Models to s3 and minio
	@aws s3 sync --delete $(HOME)/exciton/models s3://exciton-nlp/models
	@aws s3 sync --delete $(HOME)/exciton/datasets s3://exciton-nlp/datasets

package/exciton/models2minio: ## Sync Models to s3 and minio
	@mc mirror --overwrite --remove $(HOME)/exciton/models/ minio/exciton-nlp/models
	@mc mirror --overwrite --remove $(HOME)/exciton/datasets/ minio/exciton-nlp/datasets