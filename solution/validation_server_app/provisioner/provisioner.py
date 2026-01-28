import os
import time
import secrets
import string
from typing import Dict, Any, Optional

import docker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

WORKSHOP_DOMAIN = os.environ.get("WORKSHOP_DOMAIN", "127.0.0.1.nip.io")
VALIDATOR_IMAGE = os.environ.get("VALIDATOR_IMAGE", "workshop-validator-validator:latest")
INSTANCE_TTL_SECONDS = int(os.environ.get("INSTANCE_TTL_SECONDS", "3600"))
THRESHOLD = os.environ.get("THRESHOLD", "0.5")
FLAG_PREFIX = os.environ.get("FLAG_PREFIX", "FLAG{workshop_")
MAX_INSTANCES = int(os.environ.get("MAX_INSTANCES", "200"))

NETWORK_NAME = "ml-workshop_validation_server_workshop"  # must match docker-compose network
LABEL_MANAGED = "workshop.managed"
LABEL_CREATED_AT = "workshop.created_at"
LABEL_INSTANCE_ID = "workshop.instance_id"

client = docker.from_env()
app = FastAPI(title="Workshop Provisioner", version="1.0")

class ClaimResponse(BaseModel):
    instance_id: str
    submit_url: str
    submit_token: str
    expires_in_seconds: int

class ReleaseRequest(BaseModel):
    instance_id: str

def _rand_id(n=10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))

def _rand_flag(n=24) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))

def _current_managed_instances():
    return client.containers.list(
        all=True,
        filters={"label": [f"{LABEL_MANAGED}=true"]}
    )

def _cleanup_expired():
    now = int(time.time())
    for c in _current_managed_instances():
        labels = c.labels or {}
        try:
            created_at = int(labels.get(LABEL_CREATED_AT, "0"))
        except ValueError:
            created_at = 0

        if created_at and (now - created_at) > INSTANCE_TTL_SECONDS:
            try:
                c.remove(force=True)
            except Exception:
                pass

@app.get("/instances")
def list_instances():
    _cleanup_expired()
    out = []
    for c in _current_managed_instances():
        labels = c.labels or {}
        out.append({
            "name": c.name,
            "status": c.status,
            "instance_id": labels.get(LABEL_INSTANCE_ID),
            "created_at": labels.get(LABEL_CREATED_AT),
        })
    return {"count": len(out), "instances": out}

@app.post("/claim", response_model=ClaimResponse)
def claim():
    _cleanup_expired()
    instances = _current_managed_instances()
    if len(instances) >= MAX_INSTANCES:
        raise HTTPException(status_code=429, detail="Too many active instances. Try again later.")

    instance_id = _rand_id()
    submit_token = secrets.token_urlsafe(18)
    # Unique flag per student instance
    flag = f"{FLAG_PREFIX}{instance_id}_{_rand_flag()}" + "}"

    hostname = f"{instance_id}.{WORKSHOP_DOMAIN}"
    submit_url = f"http://{hostname}/submit"

    created_at = str(int(time.time()))
    container_name = f"validator-{instance_id}"

    labels = {
        LABEL_MANAGED: "true",
        LABEL_CREATED_AT: created_at,
        LABEL_INSTANCE_ID: instance_id,

        # Traefik routing
        "traefik.enable": "true",
        f"traefik.http.routers.{container_name}.rule": f"Host(`{hostname}`)",
        f"traefik.http.routers.{container_name}.entrypoints": "web",
        f"traefik.http.services.{container_name}.loadbalancer.server.port": "8000",
    }

    try:
        client.containers.run(
            image=VALIDATOR_IMAGE,
            name=container_name,
            detach=True,
            network=NETWORK_NAME,
            labels=labels,
            environment={
                "FLAG": flag,
                "THRESHOLD": THRESHOLD,
                "SUBMIT_TOKEN": submit_token,
                # Optional hard limits
                "MAX_ZIP_BYTES": str(50 * 1024 * 1024),
                "MAX_EXTRACTED_BYTES": str(200 * 1024 * 1024),
            },
            # Basic safety limits (tune as needed)
            mem_limit="2g",
            nano_cpus=int(2e9),  # ~2 CPU
        )
    except docker.errors.ImageNotFound:
        raise HTTPException(status_code=500, detail=f"Validator image not found: {VALIDATOR_IMAGE}")
    except docker.errors.APIError as e:
        raise HTTPException(status_code=500, detail=f"Failed to start validator: {e.explanation}")

    return ClaimResponse(
        instance_id=instance_id,
        submit_url=submit_url,
        submit_token=submit_token,
        expires_in_seconds=INSTANCE_TTL_SECONDS,
    )

@app.post("/release")
def release(req: ReleaseRequest):
    _cleanup_expired()
    cname = f"validator-{req.instance_id}"
    try:
        c = client.containers.get(cname)
        c.remove(force=True)
        return {"status": "released", "instance_id": req.instance_id}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Instance not found.")
