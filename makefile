all:

.PHONY: clean

clean:
	@(cd latex; make clean)
	@(cd stage1; rm -f *.msh *.vtu *.pvtu *.pvd)  # DO NOT ERASE .geo IN stage1/
	@for DIR in stage2 stage3 stage4 stage5; do \
	     (cd $$DIR; rm -f *.geo *.msh *.vtu *.pvtu *.pvd); \
	done
