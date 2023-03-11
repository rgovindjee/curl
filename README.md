# curl
*cu*riosity-based *r*einforcement *l*earning for robot systems

## Building the container

To build the container, run `docker compose build`.

Once the container is built, run it with `docker run -it --rm -p 6080:80 curl-rl-proj`.

To open a shell within the container, `docker exec -it rl_proj bash`.

In any web browser, you should be able to navigate to `localhost:6080` or `127.0.0.1:6080` and see a desktop environment.

If you are running Linux or macOS, it should also be possible to set up X11-forwarding, but this is not currently supported.
