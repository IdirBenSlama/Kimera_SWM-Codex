# KIMERA Redis Configuration Guide

This document provides comprehensive Redis configuration options for the KIMERA system, including all connection, SSL, clustering, and security parameters.

## Quick Reference

Here are all the Redis configuration parameters you requested:

```bash
# Basic Connection
redis.host=localhost
redis.port=6379
redis.username=
redis.ssl=false

# SSL/TLS Configuration
redis.ca_path=
redis.ssl_keyfile=
redis.ssl_certfile=
redis.cert_reqs=required
redis.ca_certs=

# Clustering
redis.cluster_mode=false

# Secrets
redis.password=
```

## Environment Variables

### Basic Connection Settings

```bash
# Redis server hostname or IP address
REDIS_HOST=localhost

# Redis server port
REDIS_PORT=6379

# Redis username (Redis 6.0+ ACL support)
REDIS_USERNAME=

# Redis database number (0-15)
REDIS_DB=0
```

### SSL/TLS Configuration

```bash
# Enable SSL/TLS connection
REDIS_SSL=false

# Path to SSL private key file
REDIS_SSL_KEYFILE=/path/to/redis-client.key

# Path to SSL certificate file
REDIS_SSL_CERTFILE=/path/to/redis-client.crt

# Path to SSL CA certificate file
REDIS_CA_PATH=/path/to/ca.crt

# Path to SSL CA certificates bundle
REDIS_CA_CERTS=/path/to/ca-bundle.crt

# SSL certificate requirements: none, optional, required
REDIS_CERT_REQS=required
```

### Cluster Configuration

```bash
# Enable Redis Cluster mode
REDIS_CLUSTER_MODE=false

# Redis cluster nodes (comma-separated host:port pairs)
REDIS_CLUSTER_NODES=node1:7001,node2:7002,node3:7003

# Skip full coverage check in cluster mode
REDIS_SKIP_FULL_COVERAGE_CHECK=false
```

### Performance & Connection Pool

```bash
# Maximum connections in the pool
REDIS_MAX_CONNECTIONS=50

# Socket timeout in seconds
REDIS_SOCKET_TIMEOUT=5.0

# Socket connection timeout in seconds
REDIS_SOCKET_CONNECT_TIMEOUT=5.0

# Retry commands on timeout
REDIS_RETRY_ON_TIMEOUT=true

# Health check interval in seconds (0 to disable)
REDIS_HEALTH_CHECK_INTERVAL=30

# Automatically decode responses to strings
REDIS_DECODE_RESPONSES=true

# Character encoding for Redis responses
REDIS_ENCODING=utf-8
```

### Security (Secrets)

```bash
# Redis password for authentication
REDIS_PASSWORD=your_secure_password_here
```

## Configuration Examples

### 1. Basic Local Development

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### 2. Production with SSL/TLS

```bash
REDIS_HOST=redis.production.com
REDIS_PORT=6380
REDIS_SSL=true
REDIS_SSL_CERTFILE=/etc/ssl/certs/redis-client.crt
REDIS_SSL_KEYFILE=/etc/ssl/private/redis-client.key
REDIS_CA_PATH=/etc/ssl/certs/ca.crt
REDIS_CERT_REQS=required
REDIS_USERNAME=kimera_user
REDIS_PASSWORD=your_production_password
```

### 3. Redis Cluster Setup

```bash
REDIS_CLUSTER_MODE=true
REDIS_CLUSTER_NODES=redis-1:7001,redis-2:7002,redis-3:7003
REDIS_USERNAME=cluster_user
REDIS_PASSWORD=cluster_password
REDIS_SKIP_FULL_COVERAGE_CHECK=false
```

### 4. High-Performance Configuration

```bash
REDIS_HOST=redis.internal
REDIS_PORT=6379
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_TIMEOUT=2.0
REDIS_SOCKET_CONNECT_TIMEOUT=2.0
REDIS_HEALTH_CHECK_INTERVAL=10
REDIS_RETRY_ON_TIMEOUT=true
```

## Python Code Usage

### Accessing Redis Configuration

```python
from src.config import get_settings

# Get Redis settings
settings = get_settings()
redis_config = settings.redis

# Basic connection info
print(f"Redis Host: {redis_config.host}")
print(f"Redis Port: {redis_config.port}")
print(f"SSL Enabled: {redis_config.ssl}")

# Get connection URL
connection_url = redis_config.connection_url
print(f"Connection URL: {connection_url}")

# Get connection kwargs for redis client
connection_kwargs = redis_config.connection_kwargs
```

### Creating Redis Client

```python
import redis.asyncio as aioredis
from src.config import get_settings

async def create_redis_client():
    settings = get_settings()
    
    if settings.redis.cluster_mode:
        # Create cluster client
        from redis.asyncio.cluster import RedisCluster
        client = RedisCluster(**settings.redis.connection_kwargs)
    else:
        # Create standard client
        client = aioredis.Redis(**settings.redis.connection_kwargs)
    
    return client

# Usage
redis_client = await create_redis_client()
await redis_client.ping()  # Test connection
```

## SSL Certificate Setup

### Generating Self-Signed Certificates (Development)

```bash
# Generate CA private key
openssl genrsa -out ca.key 4096

# Generate CA certificate
openssl req -new -x509 -days 365 -key ca.key -out ca.crt

# Generate Redis server private key
openssl genrsa -out redis-server.key 4096

# Generate Redis server certificate request
openssl req -new -key redis-server.key -out redis-server.csr

# Generate Redis server certificate
openssl x509 -req -days 365 -in redis-server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out redis-server.crt

# Generate client private key
openssl genrsa -out redis-client.key 4096

# Generate client certificate request
openssl req -new -key redis-client.key -out redis-client.csr

# Generate client certificate
openssl x509 -req -days 365 -in redis-client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out redis-client.crt
```

### Redis Server SSL Configuration

Add to your `redis.conf`:

```conf
# Enable TLS
port 0
tls-port 6380

# Certificate files
tls-cert-file /etc/redis/tls/redis-server.crt
tls-key-file /etc/redis/tls/redis-server.key
tls-ca-cert-file /etc/redis/tls/ca.crt

# Client certificate verification
tls-auth-clients yes
```

## Security Best Practices

### 1. Use Strong Authentication

```bash
# Enable password authentication
REDIS_PASSWORD=your_very_secure_password_123!

# Use Redis 6.0+ ACL for fine-grained permissions
REDIS_USERNAME=kimera_app_user
```

### 2. Enable SSL/TLS in Production

```bash
REDIS_SSL=true
REDIS_CERT_REQS=required
```

### 3. Network Security

- Bind Redis to specific interfaces
- Use firewall rules to restrict access
- Consider VPN or private networks

### 4. Connection Limits

```bash
# Limit connections to prevent resource exhaustion
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5.0
```

## Monitoring and Health Checks

### Health Check Configuration

```bash
# Enable automatic health checks
REDIS_HEALTH_CHECK_INTERVAL=30
```

### Connection Pool Monitoring

The Redis configuration includes built-in connection pool monitoring:

```python
# Check connection pool status
settings = get_settings()
print(f"Max connections: {settings.redis.max_connections}")
print(f"Health check interval: {settings.redis.health_check_interval}")
```

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   ```bash
   REDIS_SOCKET_TIMEOUT=10.0
   REDIS_SOCKET_CONNECT_TIMEOUT=10.0
   ```

2. **SSL Certificate Issues**
   ```bash
   # For development, use less strict requirements
   REDIS_CERT_REQS=optional
   ```

3. **Cluster Connection Issues**
   ```bash
   # Allow partial cluster coverage
   REDIS_SKIP_FULL_COVERAGE_CHECK=true
   ```

### Debugging

Enable Redis debugging in KIMERA:

```bash
KIMERA_LOG_LEVEL=DEBUG
KIMERA_FEATURES={"redis_cache": true, "redis_debug": true}
```

## Integration with KIMERA Features

### Rate Limiting

Redis is used for distributed rate limiting:

```bash
# Enable rate limiting with Redis backend
KIMERA_RATE_LIMIT_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Caching Layer

Enable Redis as part of the multi-tier caching:

```bash
KIMERA_FEATURES={"redis_cache": true}
```

### Session Storage

Use Redis for session storage:

```bash
REDIS_DB=1  # Use separate database for sessions
```

This configuration system provides full control over Redis connectivity, security, and performance for the KIMERA system. 