all:

.PHONY: clean

clean:
	@rm -f *.geo *.msh *.vtu *.pvtu *.pvd
	@(cd stage1; rm -f *.msh *.vtu *.pvtu *.pvd)  # DO NOT ERASE .geo IN stage1/
	@rm -rf __pycache__
