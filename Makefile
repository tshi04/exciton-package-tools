package/extools/build: ## DEV Build
	@cd modules/exciton-package-tools && rm -rf exciton.egg-info && python3 setup.py sdist bdist_wheel

package/extools/build_and_install: ## DEV build and install
	@cd modules/exciton-package-tools && rm -rf exciton.egg-info && python3 setup.py sdist bdist_wheel && pip install -U .

package/extools/release: guard-V ## Release the package
	@cd modules/exciton-package-tools && bump2version $(V) && git push --tags
	@cd modules/exciton-package-tools && rm -r exciton.egg-info && rm -r dist && python3 setup.py sdist bdist_wheel
	@cd modules/exciton-package-tools/dist && twine upload *.whl

package/extools/docs: ## Build documentation
	@cd modules/exciton-package-tools/docs && make clean && (rm -r source || true ) && make clean html
	@cd modules/exciton-package-tools/docs && sphinx-apidoc --force -o source ../exciton/
	@cd modules/exciton-package-tools/docs && make html
	@(rm -r /var/www/html/exciton || true) && cd modules/exciton-package-tools/docs/_build && cp -r html /var/www/html/exciton
