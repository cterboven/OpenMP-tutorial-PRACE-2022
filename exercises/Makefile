ALL_RECURSIVE=all-recursive clean-recursive 

TUT=SC19

DIRS=$(shell ls -d *-* | sed '/_/d' 2>/dev/null)

all:   $(CONFIG) all-recursive
clean: $(CONFIG) clean-recursive

$(ALL_RECURSIVE):
	@failcom='exit 1';\
   target=`echo $@ | sed s/-recursive//`; \
   for subdir in $(DIRS); do \
      (cd $$subdir && $(MAKE) $$target) || eval $$failcom; \
   done;

dist:
	@failcom='exit 1';\
   for subdir in $(DIRS); do \
      (cd $$subdir && $(MAKE) clean) || eval $$failcom; \
   done; \
   tar --exclude="*-solved*" -cvzf ../mt-openmp-$(TUT).tar.gz --transform s/mt-openmp/mt-openmp-$(TUT)/ ../mt-openmp/*

dist-solved:
	@failcom='exit 1';\
   for subdir in $(DIRS); do \
      (cd $$subdir && $(MAKE) clean) || eval $$failcom; \
   done; \
   tar -cvzf ../mt-openmp-$(TUT)-solved.tar.gz --transform s/mt-openmp/mt-openmp-$(TUT)-solved/ ../mt-openmp/* 


