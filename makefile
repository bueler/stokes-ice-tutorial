all:

.PHONY: clean

clean:
	@(cd latex; make clean)
	@(cd stage1; rm -f *.msh *.vtu *.pvtu *.pvd; rm -rf domain/)  # DO NOT ERASE .geo IN stage1/
	@for DIR in stage2 stage3 stage4 stage5 stage6; do \
	     (cd $$DIR; rm -rf __pycache__/ dome/ *.geo *.msh *.vtu *.pvtu *.pvd); \
	done
