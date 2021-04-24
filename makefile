all:

.PHONY: clean

clean:
	@rm -f *.geo *.msh *.vtu *.pvtu *.pvd
	@rm -rf __pycache__
