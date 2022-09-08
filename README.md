# protodrake

Prototyping API ideas for Drake

## One-time setup

Before building, you need to create a file named user.bazelrc in the current
directory with lines such as the following, to specify your Drake source path
and the SNOPT source path:

build --repo_env=LOCAL_DRAKE_PATH=/home/blah/blah/blah/drake
build --repo_env=SNOPT_PATH=/home/blah/blah/blah/snopt.tar.gz
