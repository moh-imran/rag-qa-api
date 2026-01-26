from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from ..models.integration import Integration
from cryptography.fernet import Fernet
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/integrations", tags=["integrations"]) 

FERNET_KEY = os.getenv("INTEGRATIONS_KEY")
if not FERNET_KEY:
    # generate a key for development if not provided (not safe for production)
    FERNET_KEY = Fernet.generate_key().decode()
fernet = Fernet(FERNET_KEY.encode())


class IntegrationCreate(BaseModel):
    name: str
    type: str
    config: Dict[str, Any]


@router.post("/", response_model=Dict[str, Any])
async def create_integration(payload: IntegrationCreate):
    """Store an integration config server-side with sensitive fields encrypted.

    Expected sensitive keys (by convention) in `config`: 'api_token', 'access_token', 'password'
    """
    try:
        cfg = dict(payload.config)
        # encrypt known sensitive fields
        for k in ('api_token', 'access_token', 'password'):
            if k in cfg and cfg[k]:
                token = str(cfg[k]).encode()
                cfg[k] = fernet.encrypt(token).decode()
        doc = Integration(name=payload.name, type=payload.type, config=cfg)
        await doc.insert()
        return {"id": str(doc.id), "name": doc.name, "type": doc.type}
    except Exception as e:
        logger.exception("Failed to create integration")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_integrations():
    docs = await Integration.find_all().to_list()
    result = []
    for d in docs:
        # Do not return encrypted tokens to clients
        cfg = dict(d.config)
        for k in ('api_token', 'access_token', 'password'):
            if k in cfg:
                cfg[k] = "***"
        result.append({"id": str(d.id), "name": d.name, "type": d.type, "config": cfg})
    return {"integrations": result}


@router.get("/{integration_id}")
async def get_integration(integration_id: str):
    doc = await Integration.get(integration_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Integration not found")
    cfg = dict(doc.config)
    # Do not expose secrets
    for k in ('api_token', 'access_token', 'password'):
        if k in cfg:
            cfg[k] = "***"
    return {"id": str(doc.id), "name": doc.name, "type": doc.type, "config": cfg}


@router.delete("/{integration_id}")
async def delete_integration(integration_id: str):
    doc = await Integration.get(integration_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Integration not found")
    await doc.delete()
    return {"status": "deleted"}


# helper to retrieve decrypted config server-side
async def get_decrypted_config(integration_id: str) -> Optional[Dict[str, Any]]:
    doc = await Integration.get(integration_id)
    if not doc:
        return None
    cfg = dict(doc.config)
    for k in ('api_token', 'access_token', 'password'):
        if k in cfg and isinstance(cfg[k], str) and cfg[k].startswith('gAAAA'):
            try:
                cfg[k] = fernet.decrypt(cfg[k].encode()).decode()
            except Exception:
                cfg[k] = None
    return cfg
