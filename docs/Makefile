SPHINXOPTS  ?= --keep-going
SOURCEDIR   = .
BUILDDIR    = _build

html:
	python3 -m sphinx -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)

apidoc:
	python3 -m sphinx.ext.apidoc -o api ../evolve -f --module-first -e
	@# Strip :members: from package-level automodule directives to prevent
	@# duplicate object descriptions from re-exported symbols.  Package RSTs
	@# will only show the module docstring; members are documented in their
	@# own leaf-module RSTs.
	@python3 -c "\
import pathlib, re; \
evolve = pathlib.Path('../evolve'); \
pkgs = {'.'.join(p.parent.relative_to(evolve.parent).parts)+'.rst' for p in evolve.rglob('__init__.py')}; \
pkgs.add('evolve.rst'); \
api = pathlib.Path('api'); \
[api.joinpath(f).write_text(re.sub(r'   :members:\n   :show-inheritance:\n   :undoc-members:\n', '   :show-inheritance:\n', api.joinpath(f).read_text())) for f in sorted(pkgs) if api.joinpath(f).exists()]"

clean:
	rm -rf $(BUILDDIR) api/

.PHONY: html apidoc clean
