SPHINXOPTS  ?= -W --keep-going
SOURCEDIR   = .
BUILDDIR    = _build

html:
	python3 -m sphinx -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)

apidoc:
	python3 -m sphinx.ext.apidoc -o api ../src/evolve_framework -f --module-first -e

clean:
	rm -rf $(BUILDDIR) api/

.PHONY: html apidoc clean
