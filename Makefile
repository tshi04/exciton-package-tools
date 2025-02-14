package/exciton_tools/build: ## DEV Build
	@cd packages/exciton-package-tools && rm -rf exciton_tools.egg-info && python3 setup.py sdist bdist_wheel

package/exciton_tools/build_and_install: ## DEV build and install
	@cd packages/exciton-package-tools && rm -rf exciton_tools.egg-info && python3 setup.py sdist bdist_wheel && pip install -U .

package/exciton_tools/release: guard-V ## Release the package
	@cd packages/exciton-package-tools && bump2version $(V) && git push --tags
	@cd packages/exciton-package-tools && (rm -r exciton_tools.egg-info || true) && (rm -r dist || true) && python3 setup.py sdist bdist_wheel
	@cd packages/exciton-package-tools/dist && twine upload *.whl

package/exciton_tools/docs: ## Build documentation
	@cd packages/exciton-package-tools/docs && make clean && (rm -r source || true ) && make clean html
	@cd packages/exciton-package-tools/docs && sphinx-apidoc --force -o source ../exciton_tools
	@cd packages/exciton-package-tools/docs && make html
	@(rm -r /var/www/html/exciton_tools || true) && cd packages/exciton-package-tools/docs/_build && cp -r html /var/www/html/exciton_tools
