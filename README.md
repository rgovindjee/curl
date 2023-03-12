# curl
**cu**riosity-based **r**einforcement **l**earning for robot systems

## Setting up Docker
Install `docker compose` (NOT `docker-compose) according to the instructions [here](https://docs.docker.com/compose/install/).

To be able to run Docker without typing `sudo`, follow the instructions [here](https://docs.docker.com/engine/install/linux-postinstall/).

## Building the container

To build the container, run `docker compose build`.

Once the container is built, run it with `docker compose up`. Use `docker compose up -d` to run in the background. 

To open a shell within the container, `docker exec -it curl-rl-proj-1 bash`. If you have multiple containers running for this project, you may have to change the name from the default.
Use `docker images` to list the built images and replace `curl-rl-proj-1` with the appropriate name. You may also press Tab to autocomplete container names if not running with `sudo`.

In any web browser, you should be able to navigate to `localhost:6080` or `127.0.0.1:6080` and see a desktop environment. 

If you are running Linux or macOS, it should also be possible to set up X11-forwarding, but this is not currently supported. 

To stop the container, use Ctrl-C or run `docker stop curl-rl-proj-1` if it is running in the background. 
