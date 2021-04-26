all:

.PHONY: clean

clean:
	@rm -f *.geo *.msh *.vtu *.pvtu *.pvd
	@(cd stage1; rm -f *.msh *.vtu *.pvtu *.pvd)  # DO NOT ERASE .geo IN stage1/
	@(cd stage2; rm -f *.geo *.msh *.vtu *.pvtu *.pvd)
	@(cd stage3; rm -f *.geo *.msh *.vtu *.pvtu *.pvd)
	@(cd stage4; rm -f *.geo *.msh *.vtu *.pvtu *.pvd)
	@rm -rf __pycache__
