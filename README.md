# curl
**cu**riosity-based **r**einforcement **l**earning for robot systems

## Setting up Docker
Install `docker compose` (NOT `docker-compose) according to the instructions [here](https://docs.docker.com/compose/install/).

To be able to run Docker without typing `sudo`, follow the instructions [here](https://docs.docker.com/engine/install/linux-postinstall/).

## Building the container

To build the container, run `docker compose build`.

Once the container is built, run it with `docker run -it --rm -p 6080:80 --name rl-proj curl-rl-proj`.

Note: if you have cloned the repository into a directory named something other than `curl/`, use `docker images` to list the built images and replace `curl-rl-proj` with the appropriate name. 

To open a shell within the container, `docker exec -it rl-proj bash`.

In any web browser, you should be able to navigate to `localhost:6080` or `127.0.0.1:6080` and see a desktop environment.

If you are running Linux or macOS, it should also be possible to set up X11-forwarding, but this is not currently supported.
