prepare-scripts:
	chmod +x run.sh

on-run: prepare-scripts

build: on-run
	./run.sh --build

run: on-run
	./run.sh
