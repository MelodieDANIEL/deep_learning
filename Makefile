# Dummy makefile to trigger the build script
all: zips
	@php build.php 

clean:
	@rm -rf web
	make -C files clean

zips:
	@make -C files/

watch:
	bash watch.sh
